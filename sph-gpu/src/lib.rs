//! GPU building blocks for one- and two-dimensional compressible SPH.
//!
//! The first implementation deliberately fixes the smoothing length and uses a
//! static cell-adjacency CSR. Particle-to-cell membership is rebuilt after each
//! move, and nested [`massively::seg::SegmentIterator`] values compose cell
//! neighborhoods with the particle segments stored in those cells.

mod config;
mod config2d;
mod error;
mod grid;
mod grid2d;
mod kernel;
mod ops;
mod ops2d;
mod particles;
mod particles2d;
mod solver;
mod solver2d;

pub use config::SolverConfig1d;
pub use config2d::SolverConfig2d;
pub use error::{Error, Result};
pub use grid::CellGraph1d;
pub use grid2d::CellGraph2d;
pub use kernel::{
    CUBIC_SPLINE_SUPPORT, cubic_spline_gradient_1d, cubic_spline_gradient_2d,
    cubic_spline_value_1d, cubic_spline_value_2d,
};
pub use particles::{DeviceParticles1d, HostParticles1d, Snapshot1d};
pub use particles2d::{DeviceParticles2d, HostParticles2d, Snapshot2d};
pub use solver::GpuSph1d;
pub use solver2d::GpuSph2d;
