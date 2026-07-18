use std::{
    error::Error,
    fs::File,
    hint::black_box,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use clap::Parser;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::Executor;
use sph_gpu::{GpuSph2d, HostParticles2d, SolverConfig2d, cubic_spline_value_2d};

#[derive(Debug, Parser)]
#[command(about = "Benchmark the 2D SPH density pipeline against CPU references")]
struct Args {
    /// Comma-separated square-lattice resolutions.
    #[arg(long, default_value = "32,64,128,256")]
    resolutions: String,

    /// Timed repetitions for the GPU and CPU cell-list implementations.
    #[arg(long, default_value_t = 5)]
    repetitions: usize,

    /// Skip the quadratic CPU reference above this particle count.
    #[arg(long, default_value_t = 16384)]
    brute_force_max: usize,

    /// CSV output path. Results are printed to stdout when omitted.
    #[arg(long)]
    out: Option<PathBuf>,
}

fn particles(resolution: usize) -> (HostParticles2d, f32) {
    let spacing = 2.0 / resolution as f32;
    let mut positions_x = Vec::with_capacity(resolution * resolution);
    let mut positions_y = Vec::with_capacity(resolution * resolution);
    for row in 0..resolution {
        let y = -1.0 + (row as f32 + 0.5) * spacing;
        for column in 0..resolution {
            positions_x.push(-1.0 + (column as f32 + 0.5) * spacing);
            positions_y.push(y);
        }
    }
    let count = positions_x.len();
    (
        HostParticles2d {
            positions_x,
            positions_y,
            velocities_x: vec![0.0; count],
            velocities_y: vec![0.0; count],
            internal_energy: vec![1.0; count],
            masses: vec![spacing * spacing; count],
        },
        spacing,
    )
}

fn density_brute_force(particles: &HostParticles2d, h: f32) -> Vec<f32> {
    let mut density = vec![0.0_f32; particles.len()];
    for (index, output) in density.iter_mut().enumerate() {
        let x = particles.positions_x[index];
        let y = particles.positions_y[index];
        let mut sum = 0.0_f32;
        for other in 0..particles.len() {
            sum += particles.masses[other]
                * cubic_spline_value_2d(
                    x - particles.positions_x[other],
                    y - particles.positions_y[other],
                    h,
                );
        }
        *output = sum;
    }
    density
}

fn density_cell_list(
    particles: &HostParticles2d,
    h: f32,
    domain_min: f32,
    domain_max: f32,
) -> Vec<f32> {
    let cell_width = 2.0 * h;
    let width = ((domain_max - domain_min) / cell_width).ceil() as usize;
    let mut cells = vec![Vec::<usize>::new(); width * width];
    let cell_coordinate =
        |position: f32| (((position - domain_min) / cell_width) as usize).min(width - 1);
    for index in 0..particles.len() {
        let cell_x = cell_coordinate(particles.positions_x[index]);
        let cell_y = cell_coordinate(particles.positions_y[index]);
        cells[cell_x + width * cell_y].push(index);
    }

    let mut density = vec![0.0_f32; particles.len()];
    for (index, output) in density.iter_mut().enumerate() {
        let x = particles.positions_x[index];
        let y = particles.positions_y[index];
        let cell_x = cell_coordinate(x);
        let cell_y = cell_coordinate(y);
        let mut sum = 0.0_f32;
        for neighbor_y in cell_y.saturating_sub(1)..=(cell_y + 1).min(width - 1) {
            for neighbor_x in cell_x.saturating_sub(1)..=(cell_x + 1).min(width - 1) {
                for &other in &cells[neighbor_x + width * neighbor_y] {
                    sum += particles.masses[other]
                        * cubic_spline_value_2d(
                            x - particles.positions_x[other],
                            y - particles.positions_y[other],
                            h,
                        );
                }
            }
        }
        *output = sum;
    }
    density
}

fn median(mut durations: Vec<Duration>) -> Duration {
    durations.sort_unstable();
    durations[durations.len() / 2]
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
    if args.repetitions == 0 {
        return Err("repetitions must be positive".into());
    }
    let resolutions = args
        .resolutions
        .split(',')
        .map(str::trim)
        .map(str::parse::<usize>)
        .collect::<Result<Vec<_>, _>>()?;
    if resolutions.is_empty() || resolutions.iter().any(|&value| value < 8) {
        return Err("every resolution must be at least 8".into());
    }

    let mut writer = output_writer(args.out.as_deref())?;
    writeln!(
        writer,
        "# resolution,particles,gpu_ms,cpu_cell_ms,cpu_bruteforce_ms,gpu_over_cpu_cell,bruteforce_over_gpu,max_relative_density_error"
    )?;
    for resolution in resolutions {
        let (host, spacing) = particles(resolution);
        let h = 1.3 * spacing;
        let domain_min = -1.0 - 2.0 * h;
        let domain_max = 1.0 + 2.0 * h;
        let solver = GpuSph2d::<WgpuRuntime>::new(
            Executor::new(WgpuDevice::DefaultDevice),
            SolverConfig2d {
                x_min: domain_min,
                x_max: domain_max,
                y_min: domain_min,
                y_max: domain_max,
                smoothing_length: h,
                ..SolverConfig2d::default()
            },
        )?;
        let mut device = solver.upload(&host)?;
        black_box(solver.snapshot(&mut device)?);

        let mut gpu_times = Vec::with_capacity(args.repetitions);
        let mut gpu_snapshot = None;
        for _ in 0..args.repetitions {
            let start = Instant::now();
            let snapshot = solver.snapshot(&mut device)?;
            gpu_times.push(start.elapsed());
            gpu_snapshot = Some(snapshot);
        }
        let gpu_time = median(gpu_times);

        black_box(density_cell_list(&host, h, domain_min, domain_max));
        let mut cpu_cell_times = Vec::with_capacity(args.repetitions);
        let mut cpu_density = Vec::new();
        for _ in 0..args.repetitions {
            let start = Instant::now();
            cpu_density = density_cell_list(&host, h, domain_min, domain_max);
            cpu_cell_times.push(start.elapsed());
            black_box(&cpu_density);
        }
        let cpu_cell_time = median(cpu_cell_times);

        let brute_force_time = if host.len() <= args.brute_force_max {
            let start = Instant::now();
            black_box(density_brute_force(&host, h));
            Some(start.elapsed())
        } else {
            None
        };

        let gpu_snapshot = gpu_snapshot.expect("at least one repetition");
        let mut max_relative_density_error = 0.0_f32;
        for index in 0..gpu_snapshot.len() {
            let expected = cpu_density[gpu_snapshot.ids[index] as usize];
            let error = (gpu_snapshot.density[index] - expected).abs() / expected.max(1.0e-12);
            max_relative_density_error = max_relative_density_error.max(error);
        }
        let gpu_ms = gpu_time.as_secs_f64() * 1.0e3;
        let cpu_cell_ms = cpu_cell_time.as_secs_f64() * 1.0e3;
        let brute_force_ms = brute_force_time
            .map(|duration| duration.as_secs_f64() * 1.0e3)
            .unwrap_or(f64::NAN);
        let gpu_over_cpu_cell = gpu_ms / cpu_cell_ms;
        let brute_force_over_gpu = brute_force_ms / gpu_ms;
        writeln!(
            writer,
            "{resolution},{},{gpu_ms:.6},{cpu_cell_ms:.6},{brute_force_ms:.6},{gpu_over_cpu_cell:.6},{brute_force_over_gpu:.6},{max_relative_density_error:.8e}",
            host.len(),
        )?;
        eprintln!(
            "benchmark: N={}, GPU={gpu_ms:.3} ms, CPU-cell={cpu_cell_ms:.3} ms, CPU-brute={brute_force_ms:.3} ms, max rel err={max_relative_density_error:.2e}",
            host.len(),
        );
    }
    writer.flush()?;
    Ok(())
}
