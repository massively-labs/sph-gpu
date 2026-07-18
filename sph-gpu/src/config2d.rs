use crate::{Error, Result};

/// Parameters shared by all particles in the fixed-smoothing-length 2D solver.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolverConfig2d {
    pub x_min: f32,
    pub x_max: f32,
    pub y_min: f32,
    pub y_max: f32,
    pub smoothing_length: f32,
    pub gamma: f32,
    pub pressure_floor: f32,
    pub internal_energy_floor: f32,
}

impl Default for SolverConfig2d {
    fn default() -> Self {
        Self {
            x_min: -1.0,
            x_max: 1.0,
            y_min: -1.0,
            y_max: 1.0,
            smoothing_length: 0.025,
            gamma: 5.0 / 3.0,
            pressure_floor: 1.0e-8,
            internal_energy_floor: 1.0e-8,
        }
    }
}

impl SolverConfig2d {
    pub fn validate(&self) -> Result<()> {
        if ![
            self.x_min,
            self.x_max,
            self.y_min,
            self.y_max,
            self.smoothing_length,
            self.gamma,
            self.pressure_floor,
            self.internal_energy_floor,
        ]
        .into_iter()
        .all(f32::is_finite)
        {
            return Err(Error::InvalidConfig(
                "all 2D parameters must be finite".into(),
            ));
        }
        if self.x_min >= self.x_max || self.y_min >= self.y_max {
            return Err(Error::InvalidConfig(
                "2D domain minima must be smaller than maxima".into(),
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
                "pressure and energy floors must be positive".into(),
            ));
        }
        let cells = self
            .cell_count_x()
            .checked_mul(self.cell_count_y())
            .ok_or_else(|| Error::InvalidConfig("2D cell count overflow".into()))?;
        if cells > u32::MAX as usize {
            return Err(Error::InvalidConfig(
                "2D cell count exceeds u32::MAX".into(),
            ));
        }
        Ok(())
    }

    pub fn support_radius(&self) -> f32 {
        2.0 * self.smoothing_length
    }

    pub fn cell_width(&self) -> f32 {
        self.support_radius()
    }

    pub fn cell_count_x(&self) -> usize {
        ((self.x_max - self.x_min) / self.cell_width()).ceil() as usize
    }

    pub fn cell_count_y(&self) -> usize {
        ((self.y_max - self.y_min) / self.cell_width()).ceil() as usize
    }
}
