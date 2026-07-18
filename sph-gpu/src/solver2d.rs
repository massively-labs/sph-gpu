use cubecl::prelude::Runtime;
use massively::{
    DeviceVec, Executor, lazy, seg::SegmentIterator, unzip2, unzip3, unzip4, unzip10, vector, zip2,
    zip3, zip4, zip5, zip7, zip8, zip10, zip11,
};

use crate::{
    CellGraph2d, DeviceParticles2d, Error, HostParticles2d, Result, Snapshot2d, SolverConfig2d,
    ops::{LessU32, ScalePosition, Thermodynamics},
    ops2d::{
        CellIndex2d, CombineDerivatives2d, Density2d, DivideMassGradientBySqrtDensity,
        InverseSqrtDensity, PressureAcceleration2d, PressureEnergyX2d, PressureEnergyY2d,
        ScaleDerivatives2d, UpdateEnergy2d, UpdateKinematics2d, ViscosityDerivative2d,
    },
};

struct PreparedState<R: Runtime> {
    cell_ids: DeviceVec<R, u32>,
    particle_offsets: DeviceVec<R, u32>,
    scaled_x: DeviceVec<R, f32>,
    scaled_y: DeviceVec<R, f32>,
    density: DeviceVec<R, f32>,
    pressure: DeviceVec<R, f32>,
    pressure_over_density2: DeviceVec<R, f32>,
}

/// Two-dimensional fixed-smoothing-length SPH executor backed by Massively.
///
/// Particles are sorted into a row-major cell grid every step. A static nine-cell
/// CSR graph and two nested [`SegmentIterator`] layers then expose candidate
/// neighbors without materializing a particle-pair list.
pub struct GpuSph2d<R: Runtime> {
    exec: Executor<R>,
    config: SolverConfig2d,
    graph: CellGraph2d,
    neighbor_cells: DeviceVec<R, u32>,
    neighbor_cell_offsets: DeviceVec<R, u32>,
}

impl<R: Runtime> GpuSph2d<R> {
    pub fn new(exec: Executor<R>, config: SolverConfig2d) -> Result<Self> {
        config.validate()?;
        let width = u32::try_from(config.cell_count_x())
            .map_err(|_| Error::InvalidConfig("2D grid width exceeds u32::MAX".into()))?;
        let height = u32::try_from(config.cell_count_y())
            .map_err(|_| Error::InvalidConfig("2D grid height exceeds u32::MAX".into()))?;
        let graph = CellGraph2d::reflecting(width, height);
        let neighbor_cells = exec.to_device(graph.neighbors());
        let neighbor_cell_offsets = exec.to_device(graph.offsets());
        Ok(Self {
            exec,
            config,
            graph,
            neighbor_cells,
            neighbor_cell_offsets,
        })
    }

    pub fn executor(&self) -> &Executor<R> {
        &self.exec
    }

    pub fn config(&self) -> &SolverConfig2d {
        &self.config
    }

    pub fn cell_graph(&self) -> &CellGraph2d {
        &self.graph
    }

    pub fn upload(&self, particles: &HostParticles2d) -> Result<DeviceParticles2d<R>> {
        particles.validate(&self.config)?;
        let inverse_h = self.config.smoothing_length.recip();
        let inverse_h2 = inverse_h * inverse_h;
        let inverse_h3 = inverse_h2 * inverse_h;
        let mass_over_h2 = particles
            .masses
            .iter()
            .map(|mass| mass * inverse_h2)
            .collect::<Vec<_>>();
        let mass_over_h3 = particles
            .masses
            .iter()
            .map(|mass| mass * inverse_h3)
            .collect::<Vec<_>>();
        let ids = (0..particles.len() as u32).collect::<Vec<_>>();

        Ok(DeviceParticles2d {
            ids: self.exec.to_device(&ids),
            positions_x: self.exec.to_device(&particles.positions_x),
            positions_y: self.exec.to_device(&particles.positions_y),
            velocities_x: self.exec.to_device(&particles.velocities_x),
            velocities_y: self.exec.to_device(&particles.velocities_y),
            internal_energy: self.exec.to_device(&particles.internal_energy),
            masses: self.exec.to_device(&particles.masses),
            mass_over_h2: self.exec.to_device(&mass_over_h2),
            mass_over_h3: self.exec.to_device(&mass_over_h3),
        })
    }

    /// Advances all particles by one first-order symplectic Euler step.
    pub fn step(&self, particles: &mut DeviceParticles2d<R>, dt: f32) -> Result<()> {
        if !dt.is_finite() || dt <= 0.0 {
            return Err(Error::InvalidConfig(
                "time step must be positive and finite".into(),
            ));
        }
        let len = particles.len();
        let prepared = self.prepare(particles)?;

        let pressure_particles_by_cell = SegmentIterator::new(
            zip4(
                prepared.scaled_x.slice(..),
                prepared.scaled_y.slice(..),
                prepared.pressure_over_density2.slice(..),
                particles.mass_over_h3.slice(..),
            ),
            prepared.particle_offsets.slice(..),
        );
        let pressure_neighbor_particle_segments = lazy::permute(
            pressure_particles_by_cell,
            lazy::transform(self.neighbor_cells.slice(..), massively::op::U32ToUsize),
        );
        let pressure_neighborhoods_by_cell = SegmentIterator::new(
            pressure_neighbor_particle_segments,
            self.neighbor_cell_offsets.slice(..),
        );
        let pressure_neighborhoods_by_particle = lazy::permute(
            pressure_neighborhoods_by_cell,
            lazy::transform(prepared.cell_ids.slice(..), massively::op::U32ToUsize),
        );
        let pressure_acceleration = vector::transform(
            &self.exec,
            zip2(
                zip3(
                    prepared.scaled_x.slice(..),
                    prepared.scaled_y.slice(..),
                    prepared.pressure_over_density2.slice(..),
                ),
                pressure_neighborhoods_by_particle,
            ),
            PressureAcceleration2d,
        )?;
        let (pressure_acceleration_x, pressure_acceleration_y) = unzip2(pressure_acceleration);

        // Pressure work is split by coordinate so every fused expression remains
        // below Massively v0.80's thirteen-read limit.
        let pressure_energy_x_particles_by_cell = SegmentIterator::new(
            zip4(
                prepared.scaled_x.slice(..),
                prepared.scaled_y.slice(..),
                particles.velocities_x.slice(..),
                particles.mass_over_h3.slice(..),
            ),
            prepared.particle_offsets.slice(..),
        );
        let pressure_energy_x_neighbor_segments = lazy::permute(
            pressure_energy_x_particles_by_cell,
            lazy::transform(self.neighbor_cells.slice(..), massively::op::U32ToUsize),
        );
        let pressure_energy_x_by_cell = SegmentIterator::new(
            pressure_energy_x_neighbor_segments,
            self.neighbor_cell_offsets.slice(..),
        );
        let pressure_energy_x_by_particle = lazy::permute(
            pressure_energy_x_by_cell,
            lazy::transform(prepared.cell_ids.slice(..), massively::op::U32ToUsize),
        );
        let pressure_energy_x = vector::transform(
            &self.exec,
            zip2(
                zip4(
                    prepared.scaled_x.slice(..),
                    prepared.scaled_y.slice(..),
                    particles.velocities_x.slice(..),
                    prepared.pressure_over_density2.slice(..),
                ),
                pressure_energy_x_by_particle,
            ),
            PressureEnergyX2d,
        )?;

        let pressure_energy_y_particles_by_cell = SegmentIterator::new(
            zip4(
                prepared.scaled_x.slice(..),
                prepared.scaled_y.slice(..),
                particles.velocities_y.slice(..),
                particles.mass_over_h3.slice(..),
            ),
            prepared.particle_offsets.slice(..),
        );
        let pressure_energy_y_neighbor_segments = lazy::permute(
            pressure_energy_y_particles_by_cell,
            lazy::transform(self.neighbor_cells.slice(..), massively::op::U32ToUsize),
        );
        let pressure_energy_y_by_cell = SegmentIterator::new(
            pressure_energy_y_neighbor_segments,
            self.neighbor_cell_offsets.slice(..),
        );
        let pressure_energy_y_by_particle = lazy::permute(
            pressure_energy_y_by_cell,
            lazy::transform(prepared.cell_ids.slice(..), massively::op::U32ToUsize),
        );
        let pressure_energy_y = vector::transform(
            &self.exec,
            zip2(
                zip4(
                    prepared.scaled_x.slice(..),
                    prepared.scaled_y.slice(..),
                    particles.velocities_y.slice(..),
                    prepared.pressure_over_density2.slice(..),
                ),
                pressure_energy_y_by_particle,
            ),
            PressureEnergyY2d,
        )?;

        let mass_gradient_over_sqrt_density = vector::transform(
            &self.exec,
            zip2(particles.mass_over_h3.slice(..), prepared.density.slice(..)),
            DivideMassGradientBySqrtDensity,
        )?;
        // Four query values, five particle values and four CSR/permute reads make
        // exactly thirteen leaves. The neighbor sqrt-density factor is precomputed;
        // the matching query factor is applied by a separate element-wise kernel.
        let viscosity_particles_by_cell = SegmentIterator::new(
            zip5(
                prepared.scaled_x.slice(..),
                prepared.scaled_y.slice(..),
                particles.velocities_x.slice(..),
                particles.velocities_y.slice(..),
                mass_gradient_over_sqrt_density.slice(..),
            ),
            prepared.particle_offsets.slice(..),
        );
        let viscosity_neighbor_segments = lazy::permute(
            viscosity_particles_by_cell,
            lazy::transform(self.neighbor_cells.slice(..), massively::op::U32ToUsize),
        );
        let viscosity_by_cell = SegmentIterator::new(
            viscosity_neighbor_segments,
            self.neighbor_cell_offsets.slice(..),
        );
        let viscosity_by_particle = lazy::permute(
            viscosity_by_cell,
            lazy::transform(prepared.cell_ids.slice(..), massively::op::U32ToUsize),
        );
        let unscaled_viscosity = vector::transform(
            &self.exec,
            zip2(
                zip4(
                    prepared.scaled_x.slice(..),
                    prepared.scaled_y.slice(..),
                    particles.velocities_x.slice(..),
                    particles.velocities_y.slice(..),
                ),
                viscosity_by_particle,
            ),
            ViscosityDerivative2d,
        )?;
        let (unscaled_viscosity_x, unscaled_viscosity_y, unscaled_viscosity_energy) =
            unzip3(unscaled_viscosity);
        let inverse_sqrt_density =
            vector::transform(&self.exec, prepared.density.slice(..), InverseSqrtDensity)?;
        let viscosity = vector::transform(
            &self.exec,
            zip4(
                unscaled_viscosity_x.slice(..),
                unscaled_viscosity_y.slice(..),
                unscaled_viscosity_energy.slice(..),
                inverse_sqrt_density.slice(..),
            ),
            ScaleDerivatives2d,
        )?;
        let (viscosity_acceleration_x, viscosity_acceleration_y, viscosity_energy) =
            unzip3(viscosity);

        let derivatives = vector::transform(
            &self.exec,
            zip7(
                pressure_acceleration_x.slice(..),
                pressure_acceleration_y.slice(..),
                pressure_energy_x.slice(..),
                pressure_energy_y.slice(..),
                viscosity_acceleration_x.slice(..),
                viscosity_acceleration_y.slice(..),
                viscosity_energy.slice(..),
            ),
            CombineDerivatives2d,
        )?;
        let (acceleration_x, acceleration_y, energy_rate) = unzip3(derivatives);

        let kinematics = vector::transform(
            &self.exec,
            zip11(
                particles.positions_x.slice(..),
                particles.positions_y.slice(..),
                particles.velocities_x.slice(..),
                particles.velocities_y.slice(..),
                acceleration_x.slice(..),
                acceleration_y.slice(..),
                lazy::constant(dt).take(len),
                lazy::constant(self.config.x_min).take(len),
                lazy::constant(self.config.x_max).take(len),
                lazy::constant(self.config.y_min).take(len),
                lazy::constant(self.config.y_max).take(len),
            ),
            UpdateKinematics2d,
        )?;
        let (positions_x, positions_y, velocities_x, velocities_y) = unzip4(kinematics);
        let internal_energy = vector::transform(
            &self.exec,
            zip4(
                particles.internal_energy.slice(..),
                energy_rate.slice(..),
                lazy::constant(dt).take(len),
                lazy::constant(self.config.internal_energy_floor).take(len),
            ),
            UpdateEnergy2d,
        )?;

        particles.positions_x = positions_x;
        particles.positions_y = positions_y;
        particles.velocities_x = velocities_x;
        particles.velocities_y = velocities_y;
        particles.internal_energy = internal_energy;
        Ok(())
    }

    pub fn step_many(
        &self,
        particles: &mut DeviceParticles2d<R>,
        dt: f32,
        steps: usize,
    ) -> Result<()> {
        for _ in 0..steps {
            self.step(particles, dt)?;
        }
        Ok(())
    }

    /// Re-sorts the state and copies a coherent density/pressure snapshot to the host.
    pub fn snapshot(&self, particles: &mut DeviceParticles2d<R>) -> Result<Snapshot2d> {
        let prepared = self.prepare(particles)?;
        Ok(Snapshot2d {
            ids: self.exec.to_host(&particles.ids)?,
            positions_x: self.exec.to_host(&particles.positions_x)?,
            positions_y: self.exec.to_host(&particles.positions_y)?,
            velocities_x: self.exec.to_host(&particles.velocities_x)?,
            velocities_y: self.exec.to_host(&particles.velocities_y)?,
            density: self.exec.to_host(&prepared.density)?,
            pressure: self.exec.to_host(&prepared.pressure)?,
            internal_energy: self.exec.to_host(&particles.internal_energy)?,
            masses: self.exec.to_host(&particles.masses)?,
        })
    }

    fn prepare(&self, particles: &mut DeviceParticles2d<R>) -> Result<PreparedState<R>> {
        let len = particles.len();
        let inverse_cell_width = self.config.cell_width().recip();
        let width = self.graph.width();
        let height = self.graph.height();
        let cell_count = self.graph.cell_count();
        let unsorted_cell_ids = vector::transform(
            &self.exec,
            zip8(
                particles.positions_x.slice(..),
                particles.positions_y.slice(..),
                lazy::constant(self.config.x_min).take(len),
                lazy::constant(self.config.y_min).take(len),
                lazy::constant(inverse_cell_width).take(len),
                lazy::constant(inverse_cell_width).take(len),
                lazy::constant(width).take(len),
                lazy::constant(height).take(len),
            ),
            CellIndex2d,
        )?;

        let sorted = vector::sort_by_key(
            &self.exec,
            unsorted_cell_ids.slice(..),
            zip10(
                unsorted_cell_ids.slice(..),
                particles.ids.slice(..),
                particles.positions_x.slice(..),
                particles.positions_y.slice(..),
                particles.velocities_x.slice(..),
                particles.velocities_y.slice(..),
                particles.internal_energy.slice(..),
                particles.masses.slice(..),
                particles.mass_over_h2.slice(..),
                particles.mass_over_h3.slice(..),
            ),
            LessU32,
        )?;
        let (
            cell_ids,
            ids,
            positions_x,
            positions_y,
            velocities_x,
            velocities_y,
            internal_energy,
            masses,
            mass_over_h2,
            mass_over_h3,
        ) = unzip10(sorted);
        particles.ids = ids;
        particles.positions_x = positions_x;
        particles.positions_y = positions_y;
        particles.velocities_x = velocities_x;
        particles.velocities_y = velocities_y;
        particles.internal_energy = internal_energy;
        particles.masses = masses;
        particles.mass_over_h2 = mass_over_h2;
        particles.mass_over_h3 = mass_over_h3;

        let particle_offsets = vector::lower_bound(
            &self.exec,
            cell_ids.slice(..),
            lazy::counting_u32(0).take(cell_count as usize + 1),
            LessU32,
        )?;
        let inverse_h = self.config.smoothing_length.recip();
        let scaled_x = vector::transform(
            &self.exec,
            zip2(
                particles.positions_x.slice(..),
                lazy::constant(inverse_h).take(len),
            ),
            ScalePosition,
        )?;
        let scaled_y = vector::transform(
            &self.exec,
            zip2(
                particles.positions_y.slice(..),
                lazy::constant(inverse_h).take(len),
            ),
            ScalePosition,
        )?;

        let density_particles_by_cell = SegmentIterator::new(
            zip3(
                scaled_x.slice(..),
                scaled_y.slice(..),
                particles.mass_over_h2.slice(..),
            ),
            particle_offsets.slice(..),
        );
        let density_neighbor_segments = lazy::permute(
            density_particles_by_cell,
            lazy::transform(self.neighbor_cells.slice(..), massively::op::U32ToUsize),
        );
        let density_by_cell = SegmentIterator::new(
            density_neighbor_segments,
            self.neighbor_cell_offsets.slice(..),
        );
        let density_by_particle = lazy::permute(
            density_by_cell,
            lazy::transform(cell_ids.slice(..), massively::op::U32ToUsize),
        );
        let density = vector::transform(
            &self.exec,
            zip2(
                zip2(scaled_x.slice(..), scaled_y.slice(..)),
                density_by_particle,
            ),
            Density2d,
        )?;
        let thermodynamics = vector::transform(
            &self.exec,
            zip4(
                density.slice(..),
                particles.internal_energy.slice(..),
                lazy::constant(self.config.gamma).take(len),
                lazy::constant(self.config.pressure_floor).take(len),
            ),
            Thermodynamics,
        )?;
        let (pressure, pressure_over_density2) = unzip2(thermodynamics);

        Ok(PreparedState {
            cell_ids,
            particle_offsets,
            scaled_x,
            scaled_y,
            density,
            pressure,
            pressure_over_density2,
        })
    }
}
