use crate::{Error, Result};

/// Parameters shared by all particles in the current 1D solver.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolverConfig1d {
    /// Lower reflecting boundary.
    pub x_min: f32,
    /// Upper reflecting boundary.
    pub x_max: f32,
    /// Fixed smoothing length. The cubic spline support radius is `2 h`.
    pub smoothing_length: f32,
    /// Ratio of specific heats used by `p = (gamma - 1) rho e`.
    pub gamma: f32,
    /// Lower pressure floor used while evaluating the equation of state.
    pub pressure_floor: f32,
    /// Lower specific-internal-energy floor applied after every step.
    pub internal_energy_floor: f32,
    /// Coefficient multiplying the linear Monaghan viscosity term.
    ///
    /// For the current global-signal-speed formulation this is normally
    /// `alpha * c_signal`.
    pub linear_viscosity: f32,
    /// Coefficient multiplying the quadratic Monaghan viscosity term.
    pub quadratic_viscosity: f32,
}

impl Default for SolverConfig1d {
    fn default() -> Self {
        Self {
            x_min: 0.0,
            x_max: 1.0,
            smoothing_length: 0.01,
            gamma: 1.4,
            pressure_floor: 1.0e-6,
            internal_energy_floor: 1.0e-6,
            linear_viscosity: 1.2,
            quadratic_viscosity: 2.0,
        }
    }
}

impl SolverConfig1d {
    pub fn validate(&self) -> Result<()> {
        let finite = [
            self.x_min,
            self.x_max,
            self.smoothing_length,
            self.gamma,
            self.pressure_floor,
            self.internal_energy_floor,
            self.linear_viscosity,
            self.quadratic_viscosity,
        ]
        .into_iter()
        .all(f32::is_finite);
        if !finite {
            return Err(Error::InvalidConfig(
                "all floating-point parameters must be finite".into(),
            ));
        }
        if self.x_min >= self.x_max {
            return Err(Error::InvalidConfig(
                "x_min must be smaller than x_max".into(),
            ));
        }
        if self.smoothing_length <= 0.0 {
            return Err(Error::InvalidConfig(
                "smoothing_length must be positive".into(),
            ));
        }
        if self.gamma <= 1.0 {
            return Err(Error::InvalidConfig(
                "gamma must be greater than one".into(),
            ));
        }
        if self.pressure_floor <= 0.0 || self.internal_energy_floor <= 0.0 {
            return Err(Error::InvalidConfig(
                "pressure and internal-energy floors must be positive".into(),
            ));
        }
        if self.linear_viscosity < 0.0 || self.quadratic_viscosity < 0.0 {
            return Err(Error::InvalidConfig(
                "artificial-viscosity coefficients must be non-negative".into(),
            ));
        }
        if self.cell_count() > u32::MAX as usize {
            return Err(Error::InvalidConfig("cell count exceeds u32::MAX".into()));
        }
        Ok(())
    }

    /// Compact support radius of the one-dimensional cubic-spline kernel.
    pub fn support_radius(&self) -> f32 {
        2.0 * self.smoothing_length
    }

    /// Cell width. Matching it to the support radius makes a three-cell stencil sufficient.
    pub fn cell_width(&self) -> f32 {
        self.support_radius()
    }

    pub fn cell_count(&self) -> usize {
        ((self.x_max - self.x_min) / self.cell_width()).ceil() as usize
    }
}
