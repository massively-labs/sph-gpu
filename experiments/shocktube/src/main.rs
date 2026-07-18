use std::{
    error::Error,
    fs::File,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

use clap::Parser;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::Executor;
use sph_gpu::{GpuSph1d, HostParticles1d, Snapshot1d, SolverConfig1d, cubic_spline_value_1d};

const GAMMA: f32 = 1.4;
const LEFT_DENSITY: f32 = 1.0;
const LEFT_PRESSURE: f32 = 1.0;
const RIGHT_DENSITY: f32 = 0.125;
const RIGHT_PRESSURE: f32 = 0.1;
const DISCONTINUITY: f32 = 0.5;

#[derive(Debug, Parser)]
#[command(about = "One-dimensional Sod shock tube solved with GPU SPH")]
struct Args {
    /// Number of right-state particles per unit length.
    #[arg(long, default_value_t = 160)]
    right_resolution: usize,

    /// Smoothing length divided by the initial right-state spacing.
    #[arg(long, default_value_t = 1.2)]
    smoothing_factor: f32,

    /// Final physical time.
    #[arg(long, default_value_t = 0.15)]
    finish: f32,

    /// Fixed time step. If omitted, a conservative CFL estimate is used.
    #[arg(long)]
    dt: Option<f32>,

    /// Number of uniformly spaced samples written over x in [0, 1].
    #[arg(long, default_value_t = 401)]
    samples: usize,

    /// Constant-state padding simulated on both sides of the plotted interval.
    #[arg(long, default_value_t = 0.25)]
    padding: f32,

    /// CSV output path. The CSV is printed to stdout when omitted.
    #[arg(long)]
    out: Option<PathBuf>,

    /// Optional text file receiving scalar error metrics.
    #[arg(long)]
    metrics: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Primitive {
    density: f32,
    velocity: f32,
    pressure: f32,
}

#[derive(Clone, Copy, Debug)]
struct SodSolution {
    gamma: f32,
    left: Primitive,
    right: Primitive,
    pressure_star: f32,
    velocity_star: f32,
}

impl SodSolution {
    fn new(gamma: f32, left: Primitive, right: Primitive) -> Self {
        let sound_left = sound_speed(gamma, left);
        let sound_right = sound_speed(gamma, right);
        let mut pressure = (0.5 * (left.pressure + right.pressure)
            - 0.125
                * (right.velocity - left.velocity)
                * (left.density + right.density)
                * (sound_left + sound_right))
            .max(1.0e-8);

        for _ in 0..32 {
            let (function_left, derivative_left) = pressure_function(gamma, pressure, left);
            let (function_right, derivative_right) = pressure_function(gamma, pressure, right);
            let next = (pressure
                - (function_left + function_right + right.velocity - left.velocity)
                    / (derivative_left + derivative_right))
                .max(1.0e-8);
            if (next - pressure).abs() <= 1.0e-7 * (next + pressure) {
                pressure = next;
                break;
            }
            pressure = next;
        }

        let (function_left, _) = pressure_function(gamma, pressure, left);
        let (function_right, _) = pressure_function(gamma, pressure, right);
        let velocity_star = 0.5 * (left.velocity + right.velocity + function_right - function_left);
        Self {
            gamma,
            left,
            right,
            pressure_star: pressure,
            velocity_star,
        }
    }

    fn sample(&self, similarity_coordinate: f32) -> Primitive {
        if similarity_coordinate <= self.velocity_star {
            self.sample_left(similarity_coordinate)
        } else {
            self.sample_right(similarity_coordinate)
        }
    }

    fn sample_left(&self, similarity_coordinate: f32) -> Primitive {
        let gamma = self.gamma;
        let state = self.left;
        let sound = sound_speed(gamma, state);
        let pressure_ratio = self.pressure_star / state.pressure;
        if self.pressure_star > state.pressure {
            let shock_speed = state.velocity
                - sound
                    * (((gamma + 1.0) / (2.0 * gamma)) * pressure_ratio
                        + (gamma - 1.0) / (2.0 * gamma))
                        .sqrt();
            if similarity_coordinate <= shock_speed {
                state
            } else {
                Primitive {
                    density: state.density * (pressure_ratio + (gamma - 1.0) / (gamma + 1.0))
                        / ((gamma - 1.0) / (gamma + 1.0) * pressure_ratio + 1.0),
                    velocity: self.velocity_star,
                    pressure: self.pressure_star,
                }
            }
        } else {
            let head_speed = state.velocity - sound;
            let star_sound = sound * pressure_ratio.powf((gamma - 1.0) / (2.0 * gamma));
            let tail_speed = self.velocity_star - star_sound;
            if similarity_coordinate <= head_speed {
                state
            } else if similarity_coordinate >= tail_speed {
                Primitive {
                    density: state.density * pressure_ratio.powf(1.0 / gamma),
                    velocity: self.velocity_star,
                    pressure: self.pressure_star,
                }
            } else {
                let velocity = 2.0 / (gamma + 1.0)
                    * (sound + 0.5 * (gamma - 1.0) * state.velocity + similarity_coordinate);
                let local_sound = 2.0 / (gamma + 1.0)
                    * (sound + 0.5 * (gamma - 1.0) * (state.velocity - similarity_coordinate));
                Primitive {
                    density: state.density * (local_sound / sound).powf(2.0 / (gamma - 1.0)),
                    velocity,
                    pressure: state.pressure
                        * (local_sound / sound).powf(2.0 * gamma / (gamma - 1.0)),
                }
            }
        }
    }

    fn sample_right(&self, similarity_coordinate: f32) -> Primitive {
        let gamma = self.gamma;
        let state = self.right;
        let sound = sound_speed(gamma, state);
        let pressure_ratio = self.pressure_star / state.pressure;
        if self.pressure_star > state.pressure {
            let shock_speed = state.velocity
                + sound
                    * (((gamma + 1.0) / (2.0 * gamma)) * pressure_ratio
                        + (gamma - 1.0) / (2.0 * gamma))
                        .sqrt();
            if similarity_coordinate >= shock_speed {
                state
            } else {
                Primitive {
                    density: state.density * (pressure_ratio + (gamma - 1.0) / (gamma + 1.0))
                        / ((gamma - 1.0) / (gamma + 1.0) * pressure_ratio + 1.0),
                    velocity: self.velocity_star,
                    pressure: self.pressure_star,
                }
            }
        } else {
            let head_speed = state.velocity + sound;
            let star_sound = sound * pressure_ratio.powf((gamma - 1.0) / (2.0 * gamma));
            let tail_speed = self.velocity_star + star_sound;
            if similarity_coordinate >= head_speed {
                state
            } else if similarity_coordinate <= tail_speed {
                Primitive {
                    density: state.density * pressure_ratio.powf(1.0 / gamma),
                    velocity: self.velocity_star,
                    pressure: self.pressure_star,
                }
            } else {
                let velocity = 2.0 / (gamma + 1.0)
                    * (-sound + 0.5 * (gamma - 1.0) * state.velocity + similarity_coordinate);
                let local_sound = 2.0 / (gamma + 1.0)
                    * (sound - 0.5 * (gamma - 1.0) * (state.velocity - similarity_coordinate));
                Primitive {
                    density: state.density * (local_sound / sound).powf(2.0 / (gamma - 1.0)),
                    velocity,
                    pressure: state.pressure
                        * (local_sound / sound).powf(2.0 * gamma / (gamma - 1.0)),
                }
            }
        }
    }
}

fn pressure_function(gamma: f32, pressure: f32, state: Primitive) -> (f32, f32) {
    if pressure > state.pressure {
        let coefficient_a = 2.0 / ((gamma + 1.0) * state.density);
        let coefficient_b = (gamma - 1.0) / (gamma + 1.0) * state.pressure;
        let root = (coefficient_a / (pressure + coefficient_b)).sqrt();
        (
            (pressure - state.pressure) * root,
            root * (1.0 - 0.5 * (pressure - state.pressure) / (pressure + coefficient_b)),
        )
    } else {
        let sound = sound_speed(gamma, state);
        let ratio = pressure / state.pressure;
        (
            2.0 * sound / (gamma - 1.0) * (ratio.powf((gamma - 1.0) / (2.0 * gamma)) - 1.0),
            ratio.powf(-(gamma + 1.0) / (2.0 * gamma)) / (state.density * sound),
        )
    }
}

fn sound_speed(gamma: f32, state: Primitive) -> f32 {
    (gamma * state.pressure / state.density).sqrt()
}

fn initial_particles(args: &Args) -> Result<(HostParticles1d, f32), Box<dyn Error>> {
    if args.right_resolution < 8 {
        return Err("right_resolution must be at least 8".into());
    }
    if !args.padding.is_finite() || args.padding <= 0.0 {
        return Err("padding must be positive and finite".into());
    }
    let x_min = -args.padding;
    let x_max = 1.0 + args.padding;
    let right_length = x_max - DISCONTINUITY;
    let left_length = DISCONTINUITY - x_min;
    let right_count = (right_length * args.right_resolution as f32).round() as usize;
    let particle_mass = RIGHT_DENSITY * right_length / right_count as f32;
    let left_count = (LEFT_DENSITY * left_length / particle_mass).round() as usize;
    let left_spacing = left_length / left_count as f32;
    let right_spacing = right_length / right_count as f32;

    let mut positions = Vec::with_capacity(left_count + right_count);
    positions.extend((0..left_count).map(|index| x_min + (index as f32 + 0.5) * left_spacing));
    positions
        .extend((0..right_count).map(|index| DISCONTINUITY + (index as f32 + 0.5) * right_spacing));
    let masses = vec![particle_mass; positions.len()];
    let velocities = vec![0.0; positions.len()];
    let left_energy = LEFT_PRESSURE / ((GAMMA - 1.0) * LEFT_DENSITY);
    let right_energy = RIGHT_PRESSURE / ((GAMMA - 1.0) * RIGHT_DENSITY);
    let internal_energy = (0..positions.len())
        .map(|index| {
            if index < left_count {
                left_energy
            } else {
                right_energy
            }
        })
        .collect();
    Ok((
        HostParticles1d {
            positions,
            velocities,
            internal_energy,
            masses,
        },
        right_spacing,
    ))
}

fn interpolated(snapshot: &Snapshot1d, x: f32, h: f32) -> (f32, f32, f32, f32) {
    let mut density = 0.0;
    let mut volume_weight = 0.0;
    let mut velocity = 0.0;
    let mut pressure = 0.0;
    let mut internal_energy = 0.0;
    for index in 0..snapshot.len() {
        let kernel = cubic_spline_value_1d(x - snapshot.positions[index], h);
        density += snapshot.masses[index] * kernel;
        let weight = snapshot.masses[index] / snapshot.density[index] * kernel;
        volume_weight += weight;
        velocity += weight * snapshot.velocities[index];
        pressure += weight * snapshot.pressure[index];
        internal_energy += weight * snapshot.internal_energy[index];
    }
    if volume_weight > 0.0 {
        (
            density,
            velocity / volume_weight,
            pressure / volume_weight,
            internal_energy / volume_weight,
        )
    } else {
        (density, 0.0, 0.0, 0.0)
    }
}

fn output_writer(path: Option<&Path>) -> Result<Box<dyn Write>, Box<dyn Error>> {
    match path {
        Some(path) => {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            Ok(Box::new(BufWriter::new(File::create(path)?)))
        }
        None => Ok(Box::new(BufWriter::new(std::io::stdout()))),
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    if !args.finish.is_finite() || args.finish <= 0.0 {
        return Err("finish must be positive and finite".into());
    }
    if !args.smoothing_factor.is_finite() || args.smoothing_factor <= 0.0 {
        return Err("smoothing_factor must be positive and finite".into());
    }
    if args.samples < 2 {
        return Err("samples must be at least two".into());
    }

    let (host_particles, right_spacing) = initial_particles(&args)?;
    let smoothing_length = args.smoothing_factor * right_spacing;
    let maximum_initial_sound_speed = (GAMMA * LEFT_PRESSURE / LEFT_DENSITY).sqrt();
    let default_dt = 0.08 * smoothing_length / maximum_initial_sound_speed;
    let requested_dt = args.dt.unwrap_or(default_dt);
    if !requested_dt.is_finite() || requested_dt <= 0.0 {
        return Err("dt must be positive and finite".into());
    }

    let config = SolverConfig1d {
        x_min: -args.padding,
        x_max: 1.0 + args.padding,
        smoothing_length,
        gamma: GAMMA,
        linear_viscosity: maximum_initial_sound_speed,
        quadratic_viscosity: 2.0,
        ..SolverConfig1d::default()
    };
    let solver = GpuSph1d::<WgpuRuntime>::new(Executor::new(WgpuDevice::DefaultDevice), config)?;
    let mut particles = solver.upload(&host_particles)?;

    let estimated_steps = (args.finish / requested_dt).ceil() as usize;
    eprintln!(
        "shocktube: particles={}, cells={}, h={smoothing_length:.6}, dt<={requested_dt:.6}, steps={estimated_steps}",
        particles.len(),
        solver.cell_graph().cell_count(),
    );
    let mut time = 0.0_f32;
    let mut step = 0usize;
    let mut next_progress = 10usize;
    while time < args.finish {
        let dt = requested_dt.min(args.finish - time);
        solver.step(&mut particles, dt)?;
        time += dt;
        step += 1;
        let progress = ((time / args.finish) * 100.0).floor() as usize;
        if progress >= next_progress {
            eprintln!("  {progress:3}% ({step}/{estimated_steps})");
            next_progress += 10;
        }
    }
    let snapshot = solver.snapshot(&mut particles)?;

    let left = Primitive {
        density: LEFT_DENSITY,
        velocity: 0.0,
        pressure: LEFT_PRESSURE,
    };
    let right = Primitive {
        density: RIGHT_DENSITY,
        velocity: 0.0,
        pressure: RIGHT_PRESSURE,
    };
    let exact = SodSolution::new(GAMMA, left, right);
    let mut writer = output_writer(args.out.as_deref())?;
    writeln!(
        writer,
        "# x,density,velocity,pressure,internal_energy,exact_density,exact_velocity,exact_pressure"
    )?;

    let mut density_l1 = 0.0_f32;
    let mut minimum_density = f32::INFINITY;
    let mut minimum_pressure = f32::INFINITY;
    for sample in 0..args.samples {
        let x = sample as f32 / (args.samples - 1) as f32;
        let (density, velocity, pressure, internal_energy) =
            interpolated(&snapshot, x, smoothing_length);
        let reference = exact.sample((x - DISCONTINUITY) / time);
        density_l1 += (density - reference.density).abs();
        minimum_density = minimum_density.min(density);
        minimum_pressure = minimum_pressure.min(pressure);
        writeln!(
            writer,
            "{x:.8},{density:.8},{velocity:.8},{pressure:.8},{internal_energy:.8},{:.8},{:.8},{:.8}",
            reference.density, reference.velocity, reference.pressure,
        )?;
    }
    writer.flush()?;
    density_l1 /= args.samples as f32;

    let initial_mass = host_particles.masses.iter().sum::<f32>();
    let final_mass = snapshot.masses.iter().sum::<f32>();
    let relative_mass_error = (final_mass - initial_mass).abs() / initial_mass;
    let metrics = format!(
        "time={time:.8}\nsteps={step}\nparticles={}\ndensity_l1={density_l1:.8}\nrelative_mass_error={relative_mass_error:.8e}\nmin_density={minimum_density:.8}\nmin_pressure={minimum_pressure:.8}\npressure_star={:.8}\nvelocity_star={:.8}\n",
        snapshot.len(),
        exact.pressure_star,
        exact.velocity_star,
    );
    eprintln!("{metrics}");
    if let Some(path) = &args.metrics {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, metrics)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sod_star_state_matches_reference_values() {
        let solution = SodSolution::new(
            1.4,
            Primitive {
                density: 1.0,
                velocity: 0.0,
                pressure: 1.0,
            },
            Primitive {
                density: 0.125,
                velocity: 0.0,
                pressure: 0.1,
            },
        );
        assert!((solution.pressure_star - 0.30313).abs() < 1.0e-4);
        assert!((solution.velocity_star - 0.92745).abs() < 1.0e-4);
    }

    #[test]
    fn exact_solution_recovers_unperturbed_far_fields() {
        let solution = SodSolution::new(
            GAMMA,
            Primitive {
                density: LEFT_DENSITY,
                velocity: 0.0,
                pressure: LEFT_PRESSURE,
            },
            Primitive {
                density: RIGHT_DENSITY,
                velocity: 0.0,
                pressure: RIGHT_PRESSURE,
            },
        );
        assert_eq!(solution.sample(-10.0), solution.left);
        assert_eq!(solution.sample(10.0), solution.right);
    }
}
