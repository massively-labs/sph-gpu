/// Static 2D CSR topology containing each cell and its at most eight neighbors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CellGraph2d {
    width: u32,
    height: u32,
    offsets: Vec<u32>,
    neighbors: Vec<u32>,
}

impl CellGraph2d {
    pub fn reflecting(width: u32, height: u32) -> Self {
        assert!(width > 0 && height > 0);
        let cell_count = width as usize * height as usize;
        let mut offsets = Vec::with_capacity(cell_count + 1);
        let mut neighbors = Vec::with_capacity(cell_count * 9);
        offsets.push(0);
        for y in 0..height {
            for x in 0..width {
                let y_begin = y.saturating_sub(1);
                let y_end = (y + 1).min(height - 1);
                let x_begin = x.saturating_sub(1);
                let x_end = (x + 1).min(width - 1);
                for neighbor_y in y_begin..=y_end {
                    for neighbor_x in x_begin..=x_end {
                        neighbors.push(neighbor_x + width * neighbor_y);
                    }
                }
                offsets.push(neighbors.len() as u32);
            }
        }
        Self {
            width,
            height,
            offsets,
            neighbors,
        }
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn cell_count(&self) -> u32 {
        self.width * self.height
    }

    pub fn offsets(&self) -> &[u32] {
        &self.offsets
    }

    pub fn neighbors(&self) -> &[u32] {
        &self.neighbors
    }

    pub fn row(&self, cell: u32) -> &[u32] {
        assert!(cell < self.cell_count());
        let begin = self.offsets[cell as usize] as usize;
        let end = self.offsets[cell as usize + 1] as usize;
        &self.neighbors[begin..end]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_dimensional_degrees_match_corner_edge_and_interior() {
        let graph = CellGraph2d::reflecting(4, 3);
        assert_eq!(graph.row(0).len(), 4);
        assert_eq!(graph.row(1).len(), 6);
        assert_eq!(graph.row(5).len(), 9);
        assert_eq!(graph.row(11).len(), 4);
    }
}
