use proptest::prelude::*;
use sph_gpu::{CellGraph1d, CellGraph2d};

proptest! {
    #[test]
    fn csr_rows_are_complete_unique_and_symmetric(cell_count in 1_u32..128) {
        let graph = CellGraph1d::reflecting(cell_count);
        let offsets = graph.offsets();

        prop_assert_eq!(offsets.len(), cell_count as usize + 1);
        prop_assert_eq!(offsets[0], 0);
        prop_assert_eq!(offsets[offsets.len() - 1] as usize, graph.neighbors().len());
        prop_assert!(offsets.windows(2).all(|pair| pair[0] <= pair[1]));

        for cell in 0..cell_count {
            let row = graph.row(cell);
            prop_assert!(row.windows(2).all(|pair| pair[0] < pair[1]));
            prop_assert!(row.contains(&cell));
            for &neighbor in row {
                prop_assert!(neighbor < cell_count);
                prop_assert!(cell.abs_diff(neighbor) <= 1);
                prop_assert!(graph.row(neighbor).contains(&cell));
            }
        }
    }
}

proptest! {
    #[test]
    fn csr_2d_rows_are_complete_unique_and_symmetric(
        width in 1_u32..16,
        height in 1_u32..16,
    ) {
        let graph = CellGraph2d::reflecting(width, height);
        let offsets = graph.offsets();
        let cell_count = width * height;

        prop_assert_eq!(offsets.len(), cell_count as usize + 1);
        prop_assert_eq!(offsets[0], 0);
        prop_assert_eq!(offsets[offsets.len() - 1] as usize, graph.neighbors().len());
        prop_assert!(offsets.windows(2).all(|pair| pair[0] <= pair[1]));

        for cell in 0..cell_count {
            let x = cell % width;
            let y = cell / width;
            let row = graph.row(cell);
            prop_assert!(row.windows(2).all(|pair| pair[0] < pair[1]));
            prop_assert!(row.contains(&cell));
            let expected_degree = (1 + usize::from(x > 0) + usize::from(x + 1 < width))
                * (1 + usize::from(y > 0) + usize::from(y + 1 < height));
            prop_assert_eq!(row.len(), expected_degree);
            for &neighbor in row {
                let neighbor_x = neighbor % width;
                let neighbor_y = neighbor / width;
                prop_assert!(neighbor_x.abs_diff(x) <= 1);
                prop_assert!(neighbor_y.abs_diff(y) <= 1);
                prop_assert!(graph.row(neighbor).contains(&cell));
            }
        }
    }
}
