use cubecl::prelude::*;
use massively::{
    Tuple2, Tuple3, Tuple4, Tuple5, Tuple7, Tuple8, Tuple11, flatten3, flatten4, flatten5,
    flatten7, flatten8, flatten11, op::UnaryOp, seg::Segment, tuple2, tuple3, tuple4,
};

pub(crate) struct CellIndex2d;

#[cubecl::cube]
impl UnaryOp<Tuple8<f32, f32, f32, f32, f32, f32, u32, u32>> for CellIndex2d {
    type Output = u32;

    fn apply(input: Tuple8<f32, f32, f32, f32, f32, f32, u32, u32>) -> u32 {
        let (x, y, x_min, y_min, inverse_cell_width_x, inverse_cell_width_y, width, height) =
            flatten8(input);
        let cell_x = if x <= x_min {
            0u32
        } else {
            (((x - x_min) * inverse_cell_width_x) as u32).min(width - 1u32)
        };
        let cell_y = if y <= y_min {
            0u32
        } else {
            (((y - y_min) * inverse_cell_width_y) as u32).min(height - 1u32)
        };
        cell_x + width * cell_y
    }
}

pub(crate) struct Density2d;

#[cubecl::cube]
impl UnaryOp<(Tuple2<f32, f32>, Segment<Segment<Tuple3<f32, f32, f32>>>)> for Density2d {
    type Output = f32;

    fn apply(input: (Tuple2<f32, f32>, Segment<Segment<Tuple3<f32, f32, f32>>>)) -> f32 {
        let scaled_x_i = input.0.0;
        let scaled_y_i = input.0.1;
        let neighborhoods = input.1;
        let outer = RuntimeCell::<u32>::new(0u32);
        let sum = RuntimeCell::<f32>::new(0.0_f32);

        while outer.read() < neighborhoods.len() {
            let particles = neighborhoods.at(outer.read());
            let inner = RuntimeCell::<u32>::new(0u32);
            while inner.read() < particles.len() {
                let (scaled_x_j, scaled_y_j, mass_over_h2_j) = flatten3(particles.at(inner.read()));
                let dx = scaled_x_i - scaled_x_j;
                let dy = scaled_y_i - scaled_y_j;
                let q = (dx * dx + dy * dy).sqrt();
                let shape = if q < 1.0_f32 {
                    1.0_f32 - 1.5_f32 * q * q + 0.75_f32 * q * q * q
                } else if q < 2.0_f32 {
                    let remaining = 2.0_f32 - q;
                    0.25_f32 * remaining * remaining * remaining
                } else {
                    0.0_f32
                };
                let normalization = 10.0_f32 / (7.0_f32 * core::f32::consts::PI);
                sum.store(sum.read() + normalization * mass_over_h2_j * shape);
                inner.store(inner.read() + 1u32);
            }
            outer.store(outer.read() + 1u32);
        }
        sum.read()
    }
}

pub(crate) struct PressureAcceleration2d;

#[cubecl::cube]
impl
    UnaryOp<(
        Tuple3<f32, f32, f32>,
        Segment<Segment<Tuple4<f32, f32, f32, f32>>>,
    )> for PressureAcceleration2d
{
    type Output = Tuple2<f32, f32>;

    fn apply(
        input: (
            Tuple3<f32, f32, f32>,
            Segment<Segment<Tuple4<f32, f32, f32, f32>>>,
        ),
    ) -> Self::Output {
        let (scaled_x_i, scaled_y_i, pressure_over_density2_i) = flatten3(input.0);
        let neighborhoods = input.1;
        let outer = RuntimeCell::<u32>::new(0u32);
        let acceleration_x = RuntimeCell::<f32>::new(0.0_f32);
        let acceleration_y = RuntimeCell::<f32>::new(0.0_f32);

        while outer.read() < neighborhoods.len() {
            let particles = neighborhoods.at(outer.read());
            let inner = RuntimeCell::<u32>::new(0u32);
            while inner.read() < particles.len() {
                let (scaled_x_j, scaled_y_j, pressure_over_density2_j, mass_over_h3_j) =
                    flatten4(particles.at(inner.read()));
                let dx = scaled_x_i - scaled_x_j;
                let dy = scaled_y_i - scaled_y_j;
                let q = (dx * dx + dy * dy).sqrt();
                if q > 0.0_f32 && q < 2.0_f32 {
                    let derivative = if q < 1.0_f32 {
                        -3.0_f32 * q + 2.25_f32 * q * q
                    } else {
                        let remaining = 2.0_f32 - q;
                        -0.75_f32 * remaining * remaining
                    };
                    let normalization = 10.0_f32 / (7.0_f32 * core::f32::consts::PI);
                    let gradient_scale = normalization * derivative / q;
                    let pair_scale = -mass_over_h3_j
                        * (pressure_over_density2_i + pressure_over_density2_j)
                        * gradient_scale;
                    acceleration_x.store(acceleration_x.read() + pair_scale * dx);
                    acceleration_y.store(acceleration_y.read() + pair_scale * dy);
                }
                inner.store(inner.read() + 1u32);
            }
            outer.store(outer.read() + 1u32);
        }

        tuple2(acceleration_x.read(), acceleration_y.read())
    }
}

pub(crate) struct PressureEnergyX2d;

#[cubecl::cube]
impl
    UnaryOp<(
        Tuple4<f32, f32, f32, f32>,
        Segment<Segment<Tuple4<f32, f32, f32, f32>>>,
    )> for PressureEnergyX2d
{
    type Output = f32;

    fn apply(
        input: (
            Tuple4<f32, f32, f32, f32>,
            Segment<Segment<Tuple4<f32, f32, f32, f32>>>,
        ),
    ) -> f32 {
        let (scaled_x_i, scaled_y_i, velocity_x_i, pressure_over_density2_i) = flatten4(input.0);
        let neighborhoods = input.1;
        let outer = RuntimeCell::<u32>::new(0u32);
        let energy_rate = RuntimeCell::<f32>::new(0.0_f32);

        while outer.read() < neighborhoods.len() {
            let particles = neighborhoods.at(outer.read());
            let inner = RuntimeCell::<u32>::new(0u32);
            while inner.read() < particles.len() {
                let (scaled_x_j, scaled_y_j, velocity_x_j, mass_over_h3_j) =
                    flatten4(particles.at(inner.read()));
                let dx = scaled_x_i - scaled_x_j;
                let dy = scaled_y_i - scaled_y_j;
                let q = (dx * dx + dy * dy).sqrt();
                if q > 0.0_f32 && q < 2.0_f32 {
                    let derivative = if q < 1.0_f32 {
                        -3.0_f32 * q + 2.25_f32 * q * q
                    } else {
                        let remaining = 2.0_f32 - q;
                        -0.75_f32 * remaining * remaining
                    };
                    let normalization = 10.0_f32 / (7.0_f32 * core::f32::consts::PI);
                    let gradient_x = normalization * derivative * dx / q;
                    energy_rate.store(
                        energy_rate.read()
                            + mass_over_h3_j
                                * pressure_over_density2_i
                                * (velocity_x_i - velocity_x_j)
                                * gradient_x,
                    );
                }
                inner.store(inner.read() + 1u32);
            }
            outer.store(outer.read() + 1u32);
        }
        energy_rate.read()
    }
}

pub(crate) struct PressureEnergyY2d;

#[cubecl::cube]
impl
    UnaryOp<(
        Tuple4<f32, f32, f32, f32>,
        Segment<Segment<Tuple4<f32, f32, f32, f32>>>,
    )> for PressureEnergyY2d
{
    type Output = f32;

    fn apply(
        input: (
            Tuple4<f32, f32, f32, f32>,
            Segment<Segment<Tuple4<f32, f32, f32, f32>>>,
        ),
    ) -> f32 {
        let (scaled_x_i, scaled_y_i, velocity_y_i, pressure_over_density2_i) = flatten4(input.0);
        let neighborhoods = input.1;
        let outer = RuntimeCell::<u32>::new(0u32);
        let energy_rate = RuntimeCell::<f32>::new(0.0_f32);

        while outer.read() < neighborhoods.len() {
            let particles = neighborhoods.at(outer.read());
            let inner = RuntimeCell::<u32>::new(0u32);
            while inner.read() < particles.len() {
                let (scaled_x_j, scaled_y_j, velocity_y_j, mass_over_h3_j) =
                    flatten4(particles.at(inner.read()));
                let dx = scaled_x_i - scaled_x_j;
                let dy = scaled_y_i - scaled_y_j;
                let q = (dx * dx + dy * dy).sqrt();
                if q > 0.0_f32 && q < 2.0_f32 {
                    let derivative = if q < 1.0_f32 {
                        -3.0_f32 * q + 2.25_f32 * q * q
                    } else {
                        let remaining = 2.0_f32 - q;
                        -0.75_f32 * remaining * remaining
                    };
                    let normalization = 10.0_f32 / (7.0_f32 * core::f32::consts::PI);
                    let gradient_y = normalization * derivative * dy / q;
                    energy_rate.store(
                        energy_rate.read()
                            + mass_over_h3_j
                                * pressure_over_density2_i
                                * (velocity_y_i - velocity_y_j)
                                * gradient_y,
                    );
                }
                inner.store(inner.read() + 1u32);
            }
            outer.store(outer.read() + 1u32);
        }
        energy_rate.read()
    }
}

pub(crate) struct DivideMassGradientBySqrtDensity;

#[cubecl::cube]
impl UnaryOp<Tuple2<f32, f32>> for DivideMassGradientBySqrtDensity {
    type Output = f32;

    fn apply(input: Tuple2<f32, f32>) -> f32 {
        input.0 / input.1.max(1.0e-12_f32).sqrt()
    }
}

pub(crate) struct InverseSqrtDensity;

#[cubecl::cube]
impl UnaryOp<f32> for InverseSqrtDensity {
    type Output = f32;

    fn apply(density: f32) -> f32 {
        1.0_f32 / density.max(1.0e-12_f32).sqrt()
    }
}

pub(crate) struct ViscosityDerivative2d;

#[cubecl::cube]
impl
    UnaryOp<(
        Tuple4<f32, f32, f32, f32>,
        Segment<Segment<Tuple5<f32, f32, f32, f32, f32>>>,
    )> for ViscosityDerivative2d
{
    type Output = Tuple3<f32, f32, f32>;

    fn apply(
        input: (
            Tuple4<f32, f32, f32, f32>,
            Segment<Segment<Tuple5<f32, f32, f32, f32, f32>>>,
        ),
    ) -> Self::Output {
        let (scaled_x_i, scaled_y_i, velocity_x_i, velocity_y_i) = flatten4(input.0);
        let neighborhoods = input.1;
        let outer = RuntimeCell::<u32>::new(0u32);
        let acceleration_x = RuntimeCell::<f32>::new(0.0_f32);
        let acceleration_y = RuntimeCell::<f32>::new(0.0_f32);
        let energy_rate = RuntimeCell::<f32>::new(0.0_f32);

        while outer.read() < neighborhoods.len() {
            let particles = neighborhoods.at(outer.read());
            let inner = RuntimeCell::<u32>::new(0u32);
            while inner.read() < particles.len() {
                let (
                    scaled_x_j,
                    scaled_y_j,
                    velocity_x_j,
                    velocity_y_j,
                    mass_over_h3_over_sqrt_density_j,
                ) = flatten5(particles.at(inner.read()));
                let dx = scaled_x_i - scaled_x_j;
                let dy = scaled_y_i - scaled_y_j;
                let velocity_dx = velocity_x_i - velocity_x_j;
                let velocity_dy = velocity_y_i - velocity_y_j;
                let q2 = dx * dx + dy * dy;
                let approach = velocity_dx * dx + velocity_dy * dy;
                if approach < 0.0_f32 && q2 > 0.0_f32 && q2 < 4.0_f32 {
                    let q = q2.sqrt();
                    let derivative = if q < 1.0_f32 {
                        -3.0_f32 * q + 2.25_f32 * q * q
                    } else {
                        let remaining = 2.0_f32 - q;
                        -0.75_f32 * remaining * remaining
                    };
                    let normalization = 10.0_f32 / (7.0_f32 * core::f32::consts::PI);
                    let gradient_scale = normalization * derivative / q;
                    let gradient_x = gradient_scale * dx;
                    let gradient_y = gradient_scale * dy;
                    let mu = approach / (q2 + 0.01_f32);
                    let artificial_pressure = -mu + 2.0_f32 * mu * mu;
                    let pair_scale = mass_over_h3_over_sqrt_density_j * artificial_pressure;
                    acceleration_x.store(acceleration_x.read() - pair_scale * gradient_x);
                    acceleration_y.store(acceleration_y.read() - pair_scale * gradient_y);
                    energy_rate.store(
                        energy_rate.read()
                            + 0.5_f32
                                * pair_scale
                                * (velocity_dx * gradient_x + velocity_dy * gradient_y),
                    );
                }
                inner.store(inner.read() + 1u32);
            }
            outer.store(outer.read() + 1u32);
        }

        tuple3(
            acceleration_x.read(),
            acceleration_y.read(),
            energy_rate.read(),
        )
    }
}

pub(crate) struct ScaleDerivatives2d;

#[cubecl::cube]
impl UnaryOp<Tuple4<f32, f32, f32, f32>> for ScaleDerivatives2d {
    type Output = Tuple3<f32, f32, f32>;

    fn apply(input: Tuple4<f32, f32, f32, f32>) -> Self::Output {
        let (acceleration_x, acceleration_y, energy_rate, scale) = flatten4(input);
        tuple3(
            acceleration_x * scale,
            acceleration_y * scale,
            energy_rate * scale,
        )
    }
}

pub(crate) struct CombineDerivatives2d;

#[cubecl::cube]
impl UnaryOp<Tuple7<f32, f32, f32, f32, f32, f32, f32>> for CombineDerivatives2d {
    type Output = Tuple3<f32, f32, f32>;

    fn apply(input: Tuple7<f32, f32, f32, f32, f32, f32, f32>) -> Self::Output {
        let (
            pressure_acceleration_x,
            pressure_acceleration_y,
            pressure_energy_x,
            pressure_energy_y,
            viscosity_acceleration_x,
            viscosity_acceleration_y,
            viscosity_energy,
        ) = flatten7(input);
        tuple3(
            pressure_acceleration_x + viscosity_acceleration_x,
            pressure_acceleration_y + viscosity_acceleration_y,
            pressure_energy_x + pressure_energy_y + viscosity_energy,
        )
    }
}

pub(crate) struct UpdateKinematics2d;

#[cubecl::cube]
impl UnaryOp<Tuple11<f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32>>
    for UpdateKinematics2d
{
    type Output = Tuple4<f32, f32, f32, f32>;

    fn apply(
        input: Tuple11<f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32>,
    ) -> Self::Output {
        let (
            position_x,
            position_y,
            velocity_x,
            velocity_y,
            acceleration_x,
            acceleration_y,
            dt,
            x_min,
            x_max,
            y_min,
            y_max,
        ) = flatten11(input);
        let mut next_velocity_x = velocity_x + dt * acceleration_x;
        let mut next_velocity_y = velocity_y + dt * acceleration_y;
        let mut next_position_x = position_x + dt * next_velocity_x;
        let mut next_position_y = position_y + dt * next_velocity_y;
        if next_position_x < x_min {
            next_position_x = 2.0_f32 * x_min - next_position_x;
            next_velocity_x = -next_velocity_x;
        }
        if next_position_x > x_max {
            next_position_x = 2.0_f32 * x_max - next_position_x;
            next_velocity_x = -next_velocity_x;
        }
        if next_position_y < y_min {
            next_position_y = 2.0_f32 * y_min - next_position_y;
            next_velocity_y = -next_velocity_y;
        }
        if next_position_y > y_max {
            next_position_y = 2.0_f32 * y_max - next_position_y;
            next_velocity_y = -next_velocity_y;
        }
        tuple4(
            next_position_x,
            next_position_y,
            next_velocity_x,
            next_velocity_y,
        )
    }
}

pub(crate) struct UpdateEnergy2d;

#[cubecl::cube]
impl UnaryOp<Tuple4<f32, f32, f32, f32>> for UpdateEnergy2d {
    type Output = f32;

    fn apply(input: Tuple4<f32, f32, f32, f32>) -> f32 {
        let (internal_energy, energy_rate, dt, internal_energy_floor) = flatten4(input);
        (internal_energy + dt * energy_rate).max(internal_energy_floor)
    }
}
