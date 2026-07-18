use cubecl::prelude::Runtime;
use massively::DeviceVec;

use crate::{Error, Result, SolverConfig1d};

/// Host-side particle data accepted by [`crate::GpuSph1d::upload`].
#[derive(Clone, Debug, PartialEq)]
pub struct HostParticles1d {
    pub positions: Vec<f32>,
    pub velocities: Vec<f32>,
    pub internal_energy: Vec<f32>,
    pub masses: Vec<f32>,
}

impl HostParticles1d {
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    pub(crate) fn validate(&self, config: &SolverConfig1d) -> Result<()> {
        let len = self.positions.len();
        if len == 0 {
            return Err(Error::InvalidParticles(
                "at least one particle is required".into(),
            ));
        }
        if self.velocities.len() != len
            || self.internal_energy.len() != len
            || self.masses.len() != len
        {
            return Err(Error::InvalidParticles(
                "position, velocity, internal-energy, and mass lengths differ".into(),
            ));
        }
        if len > u32::MAX as usize {
            return Err(Error::InvalidParticles(
                "particle count exceeds u32::MAX".into(),
            ));
        }
        for (index, (((&position, &velocity), &energy), &mass)) in self
            .positions
            .iter()
            .zip(&self.velocities)
            .zip(&self.internal_energy)
            .zip(&self.masses)
            .enumerate()
        {
            if !position.is_finite()
                || !velocity.is_finite()
                || !energy.is_finite()
                || !mass.is_finite()
            {
                return Err(Error::InvalidParticles(format!(
                    "particle {index} contains a non-finite value"
                )));
            }
            if !(config.x_min..=config.x_max).contains(&position) {
                return Err(Error::InvalidParticles(format!(
                    "particle {index} lies outside the configured domain"
                )));
            }
            if energy <= 0.0 || mass <= 0.0 {
                return Err(Error::InvalidParticles(format!(
                    "particle {index} must have positive mass and internal energy"
                )));
            }
        }
        Ok(())
    }
}

/// Device-resident structure-of-arrays particle state.
pub struct DeviceParticles1d<R: Runtime> {
    pub(crate) ids: DeviceVec<R, u32>,
    pub(crate) positions: DeviceVec<R, f32>,
    pub(crate) velocities: DeviceVec<R, f32>,
    pub(crate) internal_energy: DeviceVec<R, f32>,
    pub(crate) masses: DeviceVec<R, f32>,
    pub(crate) mass_over_h: DeviceVec<R, f32>,
    pub(crate) mass_over_h2: DeviceVec<R, f32>,
}

impl<R: Runtime> DeviceParticles1d<R> {
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }
}

/// Host snapshot evaluated at the current particle positions.
#[derive(Clone, Debug, PartialEq)]
pub struct Snapshot1d {
    pub ids: Vec<u32>,
    pub positions: Vec<f32>,
    pub velocities: Vec<f32>,
    pub density: Vec<f32>,
    pub pressure: Vec<f32>,
    pub internal_energy: Vec<f32>,
    pub masses: Vec<f32>,
}

impl Snapshot1d {
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }
}
