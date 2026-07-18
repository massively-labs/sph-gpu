use std::ops::Range;

/// Static CSR topology connecting each one-dimensional cell to itself and its neighbors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CellGraph1d {
    cell_count: u32,
    offsets: Vec<u32>,
    neighbors: Vec<u32>,
}

impl CellGraph1d {
    pub fn reflecting(cell_count: u32) -> Self {
        assert!(cell_count > 0, "a cell graph needs at least one cell");
        let mut offsets = Vec::with_capacity(cell_count as usize + 1);
        let mut neighbors = Vec::with_capacity(cell_count as usize * 3);
        offsets.push(0);

        for cell in 0..cell_count {
            let begin = cell.saturating_sub(1);
            let end = (cell + 1).min(cell_count - 1);
            neighbors.extend(begin..=end);
            offsets.push(neighbors.len() as u32);
        }

        Self {
            cell_count,
            offsets,
            neighbors,
        }
    }

    pub fn cell_count(&self) -> u32 {
        self.cell_count
    }

    pub fn offsets(&self) -> &[u32] {
        &self.offsets
    }

    pub fn neighbors(&self) -> &[u32] {
        &self.neighbors
    }

    pub fn row(&self, cell: u32) -> &[u32] {
        assert!(cell < self.cell_count);
        let range = self.row_range(cell);
        &self.neighbors[range]
    }

    fn row_range(&self, cell: u32) -> Range<usize> {
        let cell = cell as usize;
        self.offsets[cell] as usize..self.offsets[cell + 1] as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflecting_graph_has_expected_rows() {
        let graph = CellGraph1d::reflecting(4);
        assert_eq!(graph.offsets(), &[0, 2, 5, 8, 10]);
        assert_eq!(graph.row(0), &[0, 1]);
        assert_eq!(graph.row(1), &[0, 1, 2]);
        assert_eq!(graph.row(2), &[1, 2, 3]);
        assert_eq!(graph.row(3), &[2, 3]);
    }
}
