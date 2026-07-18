use cubecl::prelude::Runtime;
use massively::{
    DeviceVec, Executor, lazy, seg::SegmentIterator, unzip2, unzip3, unzip8, vector, zip2, zip3,
    zip4, zip5, zip8, zip11,
};

use crate::{
    CellGraph1d, DeviceParticles1d, Error, HostParticles1d, Result, Snapshot1d, SolverConfig1d,
    ops::{
        CellIndex1d, Density1d, LessU32, PressureDerivative1d, ScalePosition, Thermodynamics,
        UpdateParticles1d, ViscosityDerivative1d,
    },
};

struct PreparedState<R: Runtime> {
    cell_ids: DeviceVec<R, u32>,
    particle_offsets: DeviceVec<R, u32>,
    scaled_positions: DeviceVec<R, f32>,
    density: DeviceVec<R, f32>,
    pressure: DeviceVec<R, f32>,
    pressure_over_density2: DeviceVec<R, f32>,
}

/// One-dimensional compressible SPH executor backed by a Massively runtime.
pub struct GpuSph1d<R: Runtime> {
    exec: Executor<R>,
    config: SolverConfig1d,
    graph: CellGraph1d,
    neighbor_cells: DeviceVec<R, u32>,
    neighbor_cell_offsets: DeviceVec<R, u32>,
}

impl<R: Runtime> GpuSph1d<R> {
    pub fn new(exec: Executor<R>, config: SolverConfig1d) -> Result<Self> {
        config.validate()?;
        let cell_count = u32::try_from(config.cell_count())
            .map_err(|_| Error::InvalidConfig("cell count exceeds u32::MAX".into()))?;
        let graph = CellGraph1d::reflecting(cell_count);
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

    pub fn config(&self) -> &SolverConfig1d {
        &self.config
    }

    pub fn cell_graph(&self) -> &CellGraph1d {
        &self.graph
    }

    pub fn upload(&self, particles: &HostParticles1d) -> Result<DeviceParticles1d<R>> {
        particles.validate(&self.config)?;
        let inverse_h = self.config.smoothing_length.recip();
        let inverse_h2 = inverse_h * inverse_h;
        let mass_over_h = particles
            .masses
            .iter()
            .map(|mass| mass * inverse_h)
            .collect::<Vec<_>>();
        let mass_over_h2 = particles
            .masses
            .iter()
            .map(|mass| mass * inverse_h2)
            .collect::<Vec<_>>();
        let ids = (0..particles.len() as u32).collect::<Vec<_>>();

        Ok(DeviceParticles1d {
            ids: self.exec.to_device(&ids),
            positions: self.exec.to_device(&particles.positions),
            velocities: self.exec.to_device(&particles.velocities),
            internal_energy: self.exec.to_device(&particles.internal_energy),
            masses: self.exec.to_device(&particles.masses),
            mass_over_h: self.exec.to_device(&mass_over_h),
            mass_over_h2: self.exec.to_device(&mass_over_h2),
        })
    }

    /// Advances all particles by one first-order symplectic Euler step.
    pub fn step(&self, particles: &mut DeviceParticles1d<R>, dt: f32) -> Result<()> {
        if !dt.is_finite() || dt <= 0.0 {
            return Err(Error::InvalidConfig(
                "time step must be positive and finite".into(),
            ));
        }

        let prepared = self.prepare(particles)?;

        let pressure_particles_by_cell = SegmentIterator::new(
            zip4(
                prepared.scaled_positions.slice(..),
                particles.velocities.slice(..),
                prepared.pressure_over_density2.slice(..),
                particles.mass_over_h2.slice(..),
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
        let pressure_derivatives = vector::transform(
            &self.exec,
            zip2(
                zip3(
                    prepared.scaled_positions.slice(..),
                    particles.velocities.slice(..),
                    prepared.pressure_over_density2.slice(..),
                ),
                pressure_neighborhoods_by_particle,
            ),
            PressureDerivative1d,
        )?;
        let (pressure_acceleration, pressure_energy_rate) = unzip2(pressure_derivatives);

        // The nested iterator below has exactly thirteen read leaves, Massively v0.80's
        // supported maximum: five query fields, four particle fields, and four CSR/permute fields.
        let viscosity_particles_by_cell = SegmentIterator::new(
            zip4(
                prepared.scaled_positions.slice(..),
                particles.velocities.slice(..),
                prepared.density.slice(..),
                particles.mass_over_h2.slice(..),
            ),
            prepared.particle_offsets.slice(..),
        );
        let viscosity_neighbor_particle_segments = lazy::permute(
            viscosity_particles_by_cell,
            lazy::transform(self.neighbor_cells.slice(..), massively::op::U32ToUsize),
        );
        let viscosity_neighborhoods_by_cell = SegmentIterator::new(
            viscosity_neighbor_particle_segments,
            self.neighbor_cell_offsets.slice(..),
        );
        let viscosity_neighborhoods_by_particle = lazy::permute(
            viscosity_neighborhoods_by_cell,
            lazy::transform(prepared.cell_ids.slice(..), massively::op::U32ToUsize),
        );
        let viscosity_derivatives = vector::transform(
            &self.exec,
            zip2(
                zip5(
                    prepared.scaled_positions.slice(..),
                    particles.velocities.slice(..),
                    prepared.density.slice(..),
                    lazy::constant(self.config.linear_viscosity).take(particles.len()),
                    lazy::constant(self.config.quadratic_viscosity).take(particles.len()),
                ),
                viscosity_neighborhoods_by_particle,
            ),
            ViscosityDerivative1d,
        )?;
        let (viscosity_acceleration, viscosity_energy_rate) = unzip2(viscosity_derivatives);

        let updated = vector::transform(
            &self.exec,
            zip11(
                particles.positions.slice(..),
                particles.velocities.slice(..),
                particles.internal_energy.slice(..),
                pressure_acceleration.slice(..),
                pressure_energy_rate.slice(..),
                viscosity_acceleration.slice(..),
                viscosity_energy_rate.slice(..),
                lazy::constant(dt).take(particles.len()),
                lazy::constant(self.config.x_min).take(particles.len()),
                lazy::constant(self.config.x_max).take(particles.len()),
                lazy::constant(self.config.internal_energy_floor).take(particles.len()),
            ),
            UpdateParticles1d,
        )?;
        let (positions, velocities, internal_energy) = unzip3(updated);
        particles.positions = positions;
        particles.velocities = velocities;
        particles.internal_energy = internal_energy;
        Ok(())
    }

    pub fn step_many(
        &self,
        particles: &mut DeviceParticles1d<R>,
        dt: f32,
        steps: usize,
    ) -> Result<()> {
        for _ in 0..steps {
            self.step(particles, dt)?;
        }
        Ok(())
    }

    /// Re-sorts the state, evaluates density and pressure, and copies a coherent snapshot to host.
    pub fn snapshot(&self, particles: &mut DeviceParticles1d<R>) -> Result<Snapshot1d> {
        let prepared = self.prepare(particles)?;
        Ok(Snapshot1d {
            ids: self.exec.to_host(&particles.ids)?,
            positions: self.exec.to_host(&particles.positions)?,
            velocities: self.exec.to_host(&particles.velocities)?,
            density: self.exec.to_host(&prepared.density)?,
            pressure: self.exec.to_host(&prepared.pressure)?,
            internal_energy: self.exec.to_host(&particles.internal_energy)?,
            masses: self.exec.to_host(&particles.masses)?,
        })
    }

    fn prepare(&self, particles: &mut DeviceParticles1d<R>) -> Result<PreparedState<R>> {
        let len = particles.len();
        let cell_count = self.graph.cell_count();
        let inverse_cell_width = self.config.cell_width().recip();

        let unsorted_cell_ids = vector::transform(
            &self.exec,
            zip4(
                particles.positions.slice(..),
                lazy::constant(self.config.x_min).take(len),
                lazy::constant(inverse_cell_width).take(len),
                lazy::constant(cell_count).take(len),
            ),
            CellIndex1d,
        )?;

        let sorted = vector::sort_by_key(
            &self.exec,
            unsorted_cell_ids.slice(..),
            zip8(
                unsorted_cell_ids.slice(..),
                particles.ids.slice(..),
                particles.positions.slice(..),
                particles.velocities.slice(..),
                particles.internal_energy.slice(..),
                particles.masses.slice(..),
                particles.mass_over_h.slice(..),
                particles.mass_over_h2.slice(..),
            ),
            LessU32,
        )?;
        let (
            cell_ids,
            ids,
            positions,
            velocities,
            internal_energy,
            masses,
            mass_over_h,
            mass_over_h2,
        ) = unzip8(sorted);
        particles.ids = ids;
        particles.positions = positions;
        particles.velocities = velocities;
        particles.internal_energy = internal_energy;
        particles.masses = masses;
        particles.mass_over_h = mass_over_h;
        particles.mass_over_h2 = mass_over_h2;

        let particle_offsets = vector::lower_bound(
            &self.exec,
            cell_ids.slice(..),
            lazy::counting_u32(0).take(cell_count as usize + 1),
            LessU32,
        )?;
        let scaled_positions = vector::transform(
            &self.exec,
            zip2(
                particles.positions.slice(..),
                lazy::constant(self.config.smoothing_length.recip()).take(len),
            ),
            ScalePosition,
        )?;

        let density_particles_by_cell = SegmentIterator::new(
            zip2(scaled_positions.slice(..), particles.mass_over_h.slice(..)),
            particle_offsets.slice(..),
        );
        let density_neighbor_particle_segments = lazy::permute(
            density_particles_by_cell,
            lazy::transform(self.neighbor_cells.slice(..), massively::op::U32ToUsize),
        );
        let density_neighborhoods_by_cell = SegmentIterator::new(
            density_neighbor_particle_segments,
            self.neighbor_cell_offsets.slice(..),
        );
        let density_neighborhoods_by_particle = lazy::permute(
            density_neighborhoods_by_cell,
            lazy::transform(cell_ids.slice(..), massively::op::U32ToUsize),
        );
        let density = vector::transform(
            &self.exec,
            zip2(
                scaled_positions.slice(..),
                density_neighborhoods_by_particle,
            ),
            Density1d,
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
            scaled_positions,
            density,
            pressure,
            pressure_over_density2,
        })
    }
}
