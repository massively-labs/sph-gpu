use proptest::prelude::*;
use sph_gpu::{
    cubic_spline_gradient_1d, cubic_spline_gradient_2d, cubic_spline_value_1d,
    cubic_spline_value_2d,
};

proptest! {
    #[test]
    fn cubic_spline_is_even_nonnegative_and_compact(
        coordinate in -500_i32..=500,
        h_units in 1_u32..=100,
    ) {
        let h = h_units as f32 / 100.0;
        let displacement = coordinate as f32 * h / 100.0;
        let positive = cubic_spline_value_1d(displacement, h);
        let reflected = cubic_spline_value_1d(-displacement, h);

        prop_assert!(positive >= 0.0);
        prop_assert!((positive - reflected).abs() <= 2.0e-5 * positive.abs().max(1.0));
        if displacement.abs() >= 2.0 * h {
            prop_assert_eq!(positive, 0.0);
        }
    }

    #[test]
    fn cubic_spline_gradient_is_odd(
        coordinate in -199_i32..=199,
        h_units in 1_u32..=100,
    ) {
        let h = h_units as f32 / 100.0;
        let displacement = coordinate as f32 * h / 100.0;
        let gradient = cubic_spline_gradient_1d(displacement, h);
        let reflected = cubic_spline_gradient_1d(-displacement, h);

        prop_assert!((gradient + reflected).abs() <= 2.0e-4 * gradient.abs().max(1.0));
    }
}

proptest! {
    #[test]
    fn cubic_spline_2d_is_radial_nonnegative_and_compact(
        x_units in -250_i32..=250,
        y_units in -250_i32..=250,
        h_units in 1_u32..=100,
    ) {
        let h = h_units as f32 / 100.0;
        let x = x_units as f32 * h / 100.0;
        let y = y_units as f32 * h / 100.0;
        let value = cubic_spline_value_2d(x, y, h);
        let reflected = cubic_spline_value_2d(-x, -y, h);
        let rotated = cubic_spline_value_2d(-y, x, h);

        prop_assert!(value >= 0.0);
        let tolerance = 3.0e-5 * value.abs().max(1.0);
        prop_assert!((value - reflected).abs() <= tolerance);
        prop_assert!((value - rotated).abs() <= tolerance);
        if x * x + y * y >= 4.0 * h * h {
            prop_assert_eq!(value, 0.0);
        }
    }

    #[test]
    fn cubic_spline_2d_gradient_is_odd_and_radial(
        x_units in -199_i32..=199,
        y_units in -199_i32..=199,
        h_units in 1_u32..=100,
    ) {
        let h = h_units as f32 / 100.0;
        let x = x_units as f32 * h / 100.0;
        let y = y_units as f32 * h / 100.0;
        let gradient = cubic_spline_gradient_2d(x, y, h);
        let reflected = cubic_spline_gradient_2d(-x, -y, h);
        let scale = gradient.0.abs().max(gradient.1.abs()).max(1.0);

        prop_assert!((gradient.0 + reflected.0).abs() <= 3.0e-4 * scale);
        prop_assert!((gradient.1 + reflected.1).abs() <= 3.0e-4 * scale);
        prop_assert!((gradient.0 * y - gradient.1 * x).abs() <= 3.0e-4 * scale * h);
    }
}
