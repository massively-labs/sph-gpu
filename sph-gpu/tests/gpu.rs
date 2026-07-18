use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::Executor;
use sph_gpu::{
    GpuSph1d, GpuSph2d, HostParticles1d, HostParticles2d, SolverConfig1d, SolverConfig2d,
    cubic_spline_value_1d, cubic_spline_value_2d,
};

fn test_solver() -> GpuSph1d<WgpuRuntime> {
    GpuSph1d::new(
        Executor::new(WgpuDevice::Cpu),
        SolverConfig1d {
            smoothing_length: 0.1,
            linear_viscosity: 1.0,
            ..SolverConfig1d::default()
        },
    )
    .unwrap()
}

fn particles() -> HostParticles1d {
    HostParticles1d {
        positions: vec![0.12, 0.19, 0.31, 0.76],
        velocities: vec![0.0; 4],
        internal_energy: vec![2.0; 4],
        masses: vec![0.05, 0.04, 0.06, 0.05],
    }
}

#[test]
fn nested_cell_segments_match_brute_force_density() {
    let solver = test_solver();
    let host = particles();
    let mut device = solver.upload(&host).unwrap();
    let snapshot = solver.snapshot(&mut device).unwrap();
    let h = solver.config().smoothing_length;

    for (&position, &actual) in snapshot.positions.iter().zip(&snapshot.density) {
        let expected = host
            .positions
            .iter()
            .zip(&host.masses)
            .map(|(&other, &mass)| mass * cubic_spline_value_1d(position - other, h))
            .sum::<f32>();
        let tolerance = 3.0e-5 * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance,
            "density mismatch at x={position}: GPU={actual}, CPU={expected}"
        );
    }
}

#[test]
fn one_gpu_step_keeps_the_state_finite_and_inside_the_domain() {
    let solver = test_solver();
    let mut device = solver.upload(&particles()).unwrap();
    solver.step(&mut device, 1.0e-4).unwrap();
    let snapshot = solver.snapshot(&mut device).unwrap();

    for index in 0..snapshot.len() {
        assert!(snapshot.positions[index].is_finite());
        assert!(snapshot.velocities[index].is_finite());
        assert!(snapshot.internal_energy[index].is_finite());
        assert!(snapshot.density[index].is_finite());
        assert!(snapshot.pressure[index].is_finite());
        assert!(
            (solver.config().x_min..=solver.config().x_max).contains(&snapshot.positions[index])
        );
        assert!(snapshot.internal_energy[index] > 0.0);
        assert!(snapshot.density[index] > 0.0);
        assert!(snapshot.pressure[index] > 0.0);
    }
}

fn test_solver_2d() -> GpuSph2d<WgpuRuntime> {
    GpuSph2d::new(
        Executor::new(WgpuDevice::Cpu),
        SolverConfig2d {
            x_min: -1.0,
            x_max: 1.0,
            y_min: -1.0,
            y_max: 1.0,
            smoothing_length: 0.2,
            ..SolverConfig2d::default()
        },
    )
    .unwrap()
}

fn particles_2d() -> HostParticles2d {
    HostParticles2d {
        positions_x: vec![-0.42, -0.18, 0.03, 0.31, 0.44, 0.82],
        positions_y: vec![-0.31, 0.02, 0.19, -0.11, 0.35, 0.77],
        velocities_x: vec![0.02, -0.01, 0.03, -0.02, 0.0, -0.01],
        velocities_y: vec![-0.01, 0.02, 0.0, 0.01, -0.02, -0.01],
        internal_energy: vec![1.5; 6],
        masses: vec![0.09, 0.08, 0.11, 0.10, 0.07, 0.12],
    }
}

#[test]
fn nested_nine_cell_segments_match_brute_force_2d_density() {
    let solver = test_solver_2d();
    let host = particles_2d();
    let mut device = solver.upload(&host).unwrap();
    let snapshot = solver.snapshot(&mut device).unwrap();
    let h = solver.config().smoothing_length;

    for index in 0..snapshot.len() {
        let expected = host
            .positions_x
            .iter()
            .zip(&host.positions_y)
            .zip(&host.masses)
            .map(|((&x, &y), &mass)| {
                mass * cubic_spline_value_2d(
                    snapshot.positions_x[index] - x,
                    snapshot.positions_y[index] - y,
                    h,
                )
            })
            .sum::<f32>();
        let actual = snapshot.density[index];
        let tolerance = 5.0e-5 * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance,
            "2D density mismatch at ({}, {}): GPU={actual}, CPU={expected}",
            snapshot.positions_x[index],
            snapshot.positions_y[index],
        );
    }
}

#[test]
fn one_2d_gpu_step_keeps_positive_finite_state_in_domain() {
    let solver = test_solver_2d();
    let host = particles_2d();
    let initial_mass = host.masses.iter().sum::<f32>();
    let initial_momentum_x = host
        .masses
        .iter()
        .zip(&host.velocities_x)
        .map(|(&mass, &velocity)| mass * velocity)
        .sum::<f32>();
    let initial_momentum_y = host
        .masses
        .iter()
        .zip(&host.velocities_y)
        .map(|(&mass, &velocity)| mass * velocity)
        .sum::<f32>();
    let mut device = solver.upload(&host).unwrap();
    solver.step(&mut device, 1.0e-5).unwrap();
    let snapshot = solver.snapshot(&mut device).unwrap();

    for index in 0..snapshot.len() {
        assert!(snapshot.positions_x[index].is_finite());
        assert!(snapshot.positions_y[index].is_finite());
        assert!(snapshot.velocities_x[index].is_finite());
        assert!(snapshot.velocities_y[index].is_finite());
        assert!(snapshot.internal_energy[index].is_finite());
        assert!(snapshot.density[index].is_finite());
        assert!(snapshot.pressure[index].is_finite());
        assert!(
            (solver.config().x_min..=solver.config().x_max).contains(&snapshot.positions_x[index])
        );
        assert!(
            (solver.config().y_min..=solver.config().y_max).contains(&snapshot.positions_y[index])
        );
        assert!(snapshot.internal_energy[index] > 0.0);
        assert!(snapshot.density[index] > 0.0);
        assert!(snapshot.pressure[index] > 0.0);
    }

    let final_mass = snapshot.masses.iter().sum::<f32>();
    let final_momentum_x = snapshot
        .masses
        .iter()
        .zip(&snapshot.velocities_x)
        .map(|(&mass, &velocity)| mass * velocity)
        .sum::<f32>();
    let final_momentum_y = snapshot
        .masses
        .iter()
        .zip(&snapshot.velocities_y)
        .map(|(&mass, &velocity)| mass * velocity)
        .sum::<f32>();
    assert_eq!(final_mass, initial_mass);
    assert!((final_momentum_x - initial_momentum_x).abs() < 2.0e-6);
    assert!((final_momentum_y - initial_momentum_y).abs() < 2.0e-6);
}
