use std::fmt::{Display, Formatter};

/// Errors reported by the SPH setup and GPU execution layers.
#[derive(Debug)]
pub enum Error {
    InvalidConfig(String),
    InvalidParticles(String),
    Massively(massively::Error),
}

impl Display for Error {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig(message) => {
                write!(formatter, "invalid SPH configuration: {message}")
            }
            Self::InvalidParticles(message) => {
                write!(formatter, "invalid particle data: {message}")
            }
            Self::Massively(error) => Display::fmt(error, formatter),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Massively(error) => Some(error),
            Self::InvalidConfig(_) | Self::InvalidParticles(_) => None,
        }
    }
}

impl From<massively::Error> for Error {
    fn from(error: massively::Error) -> Self {
        Self::Massively(error)
    }
}

pub type Result<T> = std::result::Result<T, Error>;
