/// The cubic spline is zero for `|r| >= 2 h`.
pub const CUBIC_SPLINE_SUPPORT: f32 = 2.0;

/// One-dimensional cubic-spline kernel value.
pub fn cubic_spline_value_1d(displacement: f32, smoothing_length: f32) -> f32 {
    assert!(smoothing_length > 0.0);
    let q = displacement.abs() / smoothing_length;
    let shape = if q < 1.0 {
        1.0 - 1.5 * q * q + 0.75 * q * q * q
    } else if q < 2.0 {
        let remaining = 2.0 - q;
        0.25 * remaining * remaining * remaining
    } else {
        0.0
    };
    (2.0 / 3.0) * shape / smoothing_length
}

/// Derivative of the one-dimensional cubic spline with respect to the first position.
pub fn cubic_spline_gradient_1d(displacement: f32, smoothing_length: f32) -> f32 {
    assert!(smoothing_length > 0.0);
    let q = displacement.abs() / smoothing_length;
    let derivative = if q < 1.0 {
        -3.0 * q + 2.25 * q * q
    } else if q < 2.0 {
        let remaining = 2.0 - q;
        -0.75 * remaining * remaining
    } else {
        0.0
    };
    let sign = displacement.signum();
    (2.0 / 3.0) * derivative * sign / (smoothing_length * smoothing_length)
}

/// Two-dimensional cubic-spline kernel value with normalization `10 / (7 pi h^2)`.
pub fn cubic_spline_value_2d(dx: f32, dy: f32, smoothing_length: f32) -> f32 {
    assert!(smoothing_length > 0.0);
    let q = (dx * dx + dy * dy).sqrt() / smoothing_length;
    let shape = if q < 1.0 {
        1.0 - 1.5 * q * q + 0.75 * q * q * q
    } else if q < 2.0 {
        let remaining = 2.0 - q;
        0.25 * remaining * remaining * remaining
    } else {
        0.0
    };
    10.0 / (7.0 * std::f32::consts::PI) * shape / (smoothing_length * smoothing_length)
}

/// Gradient of the 2D cubic spline with respect to the first position.
pub fn cubic_spline_gradient_2d(dx: f32, dy: f32, smoothing_length: f32) -> (f32, f32) {
    assert!(smoothing_length > 0.0);
    let radius = (dx * dx + dy * dy).sqrt();
    if radius == 0.0 {
        return (0.0, 0.0);
    }
    let q = radius / smoothing_length;
    let derivative = if q < 1.0 {
        -3.0 * q + 2.25 * q * q
    } else if q < 2.0 {
        let remaining = 2.0 - q;
        -0.75 * remaining * remaining
    } else {
        0.0
    };
    let scale = 10.0 / (7.0 * std::f32::consts::PI) * derivative
        / (smoothing_length * smoothing_length * smoothing_length * q);
    (scale * dx / smoothing_length, scale * dy / smoothing_length)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cubic_spline_has_compact_support() {
        let h = 0.2;
        assert_eq!(cubic_spline_value_1d(2.0 * h, h), 0.0);
        assert_eq!(cubic_spline_value_1d(-3.0 * h, h), 0.0);
        assert_eq!(cubic_spline_gradient_1d(2.0 * h, h), 0.0);
    }

    #[test]
    fn cubic_spline_gradient_is_antisymmetric() {
        let h = 0.2;
        for displacement in [0.01, 0.1, 0.25, 0.39] {
            let left = cubic_spline_gradient_1d(displacement, h);
            let right = cubic_spline_gradient_1d(-displacement, h);
            assert!((left + right).abs() < 1.0e-6);
        }
    }
}
