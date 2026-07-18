use cubecl::prelude::Runtime;
use massively::DeviceVec;

use crate::{Error, Result, SolverConfig2d};

#[derive(Clone, Debug, PartialEq)]
pub struct HostParticles2d {
    pub positions_x: Vec<f32>,
    pub positions_y: Vec<f32>,
    pub velocities_x: Vec<f32>,
    pub velocities_y: Vec<f32>,
    pub internal_energy: Vec<f32>,
    pub masses: Vec<f32>,
}

impl HostParticles2d {
    pub fn len(&self) -> usize {
        self.positions_x.len()
    }

    pub fn is_empty(&self) -> bool {
        self.positions_x.is_empty()
    }

    pub(crate) fn validate(&self, config: &SolverConfig2d) -> Result<()> {
        let len = self.len();
        if len == 0 {
            return Err(Error::InvalidParticles(
                "at least one 2D particle is required".into(),
            ));
        }
        if [
            self.positions_y.len(),
            self.velocities_x.len(),
            self.velocities_y.len(),
            self.internal_energy.len(),
            self.masses.len(),
        ]
        .into_iter()
        .any(|other| other != len)
        {
            return Err(Error::InvalidParticles(
                "2D particle column lengths differ".into(),
            ));
        }
        if len > u32::MAX as usize {
            return Err(Error::InvalidParticles(
                "2D particle count exceeds u32::MAX".into(),
            ));
        }
        for index in 0..len {
            let values = [
                self.positions_x[index],
                self.positions_y[index],
                self.velocities_x[index],
                self.velocities_y[index],
                self.internal_energy[index],
                self.masses[index],
            ];
            if !values.into_iter().all(f32::is_finite) {
                return Err(Error::InvalidParticles(format!(
                    "2D particle {index} contains a non-finite value"
                )));
            }
            if !(config.x_min..=config.x_max).contains(&self.positions_x[index])
                || !(config.y_min..=config.y_max).contains(&self.positions_y[index])
            {
                return Err(Error::InvalidParticles(format!(
                    "2D particle {index} lies outside the domain"
                )));
            }
            if self.internal_energy[index] <= 0.0 || self.masses[index] <= 0.0 {
                return Err(Error::InvalidParticles(format!(
                    "2D particle {index} must have positive mass and energy"
                )));
            }
        }
        Ok(())
    }
}

pub struct DeviceParticles2d<R: Runtime> {
    pub(crate) ids: DeviceVec<R, u32>,
    pub(crate) positions_x: DeviceVec<R, f32>,
    pub(crate) positions_y: DeviceVec<R, f32>,
    pub(crate) velocities_x: DeviceVec<R, f32>,
    pub(crate) velocities_y: DeviceVec<R, f32>,
    pub(crate) internal_energy: DeviceVec<R, f32>,
    pub(crate) masses: DeviceVec<R, f32>,
    pub(crate) mass_over_h2: DeviceVec<R, f32>,
    pub(crate) mass_over_h3: DeviceVec<R, f32>,
}

impl<R: Runtime> DeviceParticles2d<R> {
    pub fn len(&self) -> usize {
        self.positions_x.len()
    }

    pub fn is_empty(&self) -> bool {
        self.positions_x.is_empty()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Snapshot2d {
    pub ids: Vec<u32>,
    pub positions_x: Vec<f32>,
    pub positions_y: Vec<f32>,
    pub velocities_x: Vec<f32>,
    pub velocities_y: Vec<f32>,
    pub density: Vec<f32>,
    pub pressure: Vec<f32>,
    pub internal_energy: Vec<f32>,
    pub masses: Vec<f32>,
}

impl Snapshot2d {
    pub fn len(&self) -> usize {
        self.positions_x.len()
    }

    pub fn is_empty(&self) -> bool {
        self.positions_x.is_empty()
    }
}
