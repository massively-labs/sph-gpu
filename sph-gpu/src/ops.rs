use cubecl::prelude::*;
use massively::{
    Tuple3, Tuple4, Tuple5, Tuple11, flatten3, flatten4, flatten5, flatten11,
    op::{BinaryPredicateOp, UnaryOp},
    seg::Segment,
    tuple2, tuple3,
};

pub(crate) struct LessU32;

#[cubecl::cube]
impl BinaryPredicateOp<u32> for LessU32 {
    fn apply(lhs: u32, rhs: u32) -> bool {
        lhs < rhs
    }
}

pub(crate) struct CellIndex1d;

#[cubecl::cube]
impl UnaryOp<Tuple4<f32, f32, f32, u32>> for CellIndex1d {
    type Output = u32;

    fn apply(input: Tuple4<f32, f32, f32, u32>) -> u32 {
        let (position, x_min, inverse_cell_width, cell_count) = flatten4(input);
        if position <= x_min {
            0u32
        } else {
            let cell = ((position - x_min) * inverse_cell_width) as u32;
            cell.min(cell_count - 1u32)
        }
    }
}

pub(crate) struct ScalePosition;

#[cubecl::cube]
impl UnaryOp<(f32, f32)> for ScalePosition {
    type Output = f32;

    fn apply(input: (f32, f32)) -> f32 {
        input.0 * input.1
    }
}

pub(crate) struct Thermodynamics;

#[cubecl::cube]
impl UnaryOp<Tuple4<f32, f32, f32, f32>> for Thermodynamics {
    type Output = (f32, f32);

    fn apply(input: Tuple4<f32, f32, f32, f32>) -> Self::Output {
        let (density, internal_energy, gamma, pressure_floor) = flatten4(input);
        let safe_density = density.max(1.0e-12_f32);
        let pressure = ((gamma - 1.0_f32) * safe_density * internal_energy).max(pressure_floor);
        let pressure_over_density2 = pressure / (safe_density * safe_density);
        tuple2(pressure, pressure_over_density2)
    }
}

pub(crate) struct Density1d;

#[cubecl::cube]
impl UnaryOp<(f32, Segment<Segment<(f32, f32)>>)> for Density1d {
    type Output = f32;

    fn apply(input: (f32, Segment<Segment<(f32, f32)>>)) -> f32 {
        let scaled_position_i = input.0;
        let neighborhoods = input.1;
        let outer = RuntimeCell::<u32>::new(0u32);
        let sum = RuntimeCell::<f32>::new(0.0_f32);

        while outer.read() < neighborhoods.len() {
            let particles = neighborhoods.at(outer.read());
            let inner = RuntimeCell::<u32>::new(0u32);
            while inner.read() < particles.len() {
                let particle = particles.at(inner.read());
                let scaled_distance = (scaled_position_i - particle.0).abs();
                let shape = if scaled_distance < 1.0_f32 {
                    1.0_f32 - 1.5_f32 * scaled_distance * scaled_distance
                        + 0.75_f32 * scaled_distance * scaled_distance * scaled_distance
                } else if scaled_distance < 2.0_f32 {
                    let remaining = 2.0_f32 - scaled_distance;
                    0.25_f32 * remaining * remaining * remaining
                } else {
                    0.0_f32
                };
                sum.store(sum.read() + (2.0_f32 / 3.0_f32) * particle.1 * shape);
                inner.store(inner.read() + 1u32);
            }
            outer.store(outer.read() + 1u32);
        }
        sum.read()
    }
}

pub(crate) struct PressureDerivative1d;

#[cubecl::cube]
impl
    UnaryOp<(
        Tuple3<f32, f32, f32>,
        Segment<Segment<Tuple4<f32, f32, f32, f32>>>,
    )> for PressureDerivative1d
{
    type Output = (f32, f32);

    fn apply(
        input: (
            Tuple3<f32, f32, f32>,
            Segment<Segment<Tuple4<f32, f32, f32, f32>>>,
        ),
    ) -> Self::Output {
        let (scaled_position_i, velocity_i, pressure_over_density2_i) = flatten3(input.0);
        let neighborhoods = input.1;
        let outer = RuntimeCell::<u32>::new(0u32);
        let acceleration = RuntimeCell::<f32>::new(0.0_f32);
        let energy_rate = RuntimeCell::<f32>::new(0.0_f32);

        while outer.read() < neighborhoods.len() {
            let particles = neighborhoods.at(outer.read());
            let inner = RuntimeCell::<u32>::new(0u32);
            while inner.read() < particles.len() {
                let particle = particles.at(inner.read());
                let (scaled_position_j, velocity_j, pressure_over_density2_j, mass_over_h2_j) =
                    flatten4(particle);
                let scaled_displacement = scaled_position_i - scaled_position_j;
                let q = scaled_displacement.abs();
                let derivative = if q < 1.0_f32 {
                    -3.0_f32 * q + 2.25_f32 * q * q
                } else if q < 2.0_f32 {
                    let remaining = 2.0_f32 - q;
                    -0.75_f32 * remaining * remaining
                } else {
                    0.0_f32
                };
                let sign = if scaled_displacement < 0.0_f32 {
                    -1.0_f32
                } else if scaled_displacement > 0.0_f32 {
                    1.0_f32
                } else {
                    0.0_f32
                };
                let gradient_shape = (2.0_f32 / 3.0_f32) * derivative * sign;
                let pair_acceleration = -mass_over_h2_j
                    * (pressure_over_density2_i + pressure_over_density2_j)
                    * gradient_shape;
                let pair_energy = mass_over_h2_j
                    * pressure_over_density2_i
                    * (velocity_i - velocity_j)
                    * gradient_shape;
                acceleration.store(acceleration.read() + pair_acceleration);
                energy_rate.store(energy_rate.read() + pair_energy);
                inner.store(inner.read() + 1u32);
            }
            outer.store(outer.read() + 1u32);
        }

        tuple2(acceleration.read(), energy_rate.read())
    }
}

pub(crate) struct ViscosityDerivative1d;

#[cubecl::cube]
impl
    UnaryOp<(
        Tuple5<f32, f32, f32, f32, f32>,
        Segment<Segment<Tuple4<f32, f32, f32, f32>>>,
    )> for ViscosityDerivative1d
{
    type Output = (f32, f32);

    fn apply(
        input: (
            Tuple5<f32, f32, f32, f32, f32>,
            Segment<Segment<Tuple4<f32, f32, f32, f32>>>,
        ),
    ) -> Self::Output {
        let (scaled_position_i, velocity_i, density_i, linear_viscosity, quadratic_viscosity) =
            flatten5(input.0);
        let neighborhoods = input.1;
        let outer = RuntimeCell::<u32>::new(0u32);
        let acceleration = RuntimeCell::<f32>::new(0.0_f32);
        let energy_rate = RuntimeCell::<f32>::new(0.0_f32);

        while outer.read() < neighborhoods.len() {
            let particles = neighborhoods.at(outer.read());
            let inner = RuntimeCell::<u32>::new(0u32);
            while inner.read() < particles.len() {
                let particle = particles.at(inner.read());
                let (scaled_position_j, velocity_j, density_j, mass_over_h2_j) = flatten4(particle);
                let scaled_displacement = scaled_position_i - scaled_position_j;
                let velocity_difference = velocity_i - velocity_j;
                let q = scaled_displacement.abs();
                let derivative = if q < 1.0_f32 {
                    -3.0_f32 * q + 2.25_f32 * q * q
                } else if q < 2.0_f32 {
                    let remaining = 2.0_f32 - q;
                    -0.75_f32 * remaining * remaining
                } else {
                    0.0_f32
                };
                let sign = if scaled_displacement < 0.0_f32 {
                    -1.0_f32
                } else if scaled_displacement > 0.0_f32 {
                    1.0_f32
                } else {
                    0.0_f32
                };
                let gradient_shape = (2.0_f32 / 3.0_f32) * derivative * sign;

                if velocity_difference * scaled_displacement < 0.0_f32 && q < 2.0_f32 {
                    let mu = velocity_difference * scaled_displacement
                        / (scaled_displacement * scaled_displacement + 0.01_f32);
                    let mean_density = 0.5_f32 * (density_i + density_j).max(1.0e-12_f32);
                    let artificial_pressure =
                        (-linear_viscosity * mu + quadratic_viscosity * mu * mu) / mean_density;
                    let pair_acceleration = -mass_over_h2_j * artificial_pressure * gradient_shape;
                    let pair_energy = 0.5_f32
                        * mass_over_h2_j
                        * artificial_pressure
                        * velocity_difference
                        * gradient_shape;
                    acceleration.store(acceleration.read() + pair_acceleration);
                    energy_rate.store(energy_rate.read() + pair_energy);
                }
                inner.store(inner.read() + 1u32);
            }
            outer.store(outer.read() + 1u32);
        }

        tuple2(acceleration.read(), energy_rate.read())
    }
}

pub(crate) struct UpdateParticles1d;

#[cubecl::cube]
impl UnaryOp<Tuple11<f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32>> for UpdateParticles1d {
    type Output = Tuple3<f32, f32, f32>;

    fn apply(
        input: Tuple11<f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32>,
    ) -> Self::Output {
        let (
            position,
            velocity,
            internal_energy,
            pressure_acceleration,
            pressure_energy_rate,
            viscosity_acceleration,
            viscosity_energy_rate,
            dt,
            x_min,
            x_max,
            internal_energy_floor,
        ) = flatten11(input);

        let acceleration = pressure_acceleration + viscosity_acceleration;
        let mut next_velocity = velocity + dt * acceleration;
        let mut next_position = position + dt * next_velocity;
        let next_internal_energy = (internal_energy
            + dt * (pressure_energy_rate + viscosity_energy_rate))
            .max(internal_energy_floor);

        if next_position < x_min {
            next_position = 2.0_f32 * x_min - next_position;
            next_velocity = -next_velocity;
        }
        if next_position > x_max {
            next_position = 2.0_f32 * x_max - next_position;
            next_velocity = -next_velocity;
        }

        tuple3(next_position, next_velocity, next_internal_energy)
    }
}
