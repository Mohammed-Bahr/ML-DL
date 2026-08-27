use crate::vec2::Vec2;

/// An oval track: an ellipse ring around a grass infield.
///
/// Pure geometry only — drawing happens in main.rs.
pub struct Track {
    pub center: Vec2,
    pub rx: f32, // horizontal radius of the outer edge
    pub ry: f32, // vertical radius of the outer edge
}

impl Track {
    /// Road width, as a fraction of the outer radius (INNER..1.0 is asphalt).
    pub const INNER: f32 = 0.72;

    pub fn new() -> Self {
        Self {
            center: Vec2::new(600.0, 350.0), // matches SCREEN_W/H in main.rs
            rx: 480.0,
            ry: 240.0,
        }
    }

    /// Normalized distance from the track center:
    /// 0 = middle of the infield, INNER = inner wall, 1.0 = outer wall.
    fn radial(&self, p: Vec2) -> f32 {
        let dx = (p.x - self.center.x) / self.rx;
        let dy = (p.y - self.center.y) / self.ry;
        (dx * dx + dy * dy).sqrt()
    }

    /// Is this world position on the asphalt? (not inside walls)
    pub fn on_track(&self, p: Vec2) -> bool {
        let s = self.radial(p);
        s >= Self::INNER && s <= 1.0
    }

    /// The car's angular position around the track, in radians [0, TAU).
    pub fn angle_of(&self, p: Vec2) -> f32 {
        let dx = (p.x - self.center.x) / self.rx;
        let dy = (p.y - self.center.y) / self.ry;
        dy.atan2(dx).rem_euclid(std::f32::consts::TAU)
    }

    /// Border collision: if `p` crossed a wall, return it pushed back
    /// onto the border, otherwise None.
    pub fn clamp_to_road(&self, p: Vec2) -> Option<Vec2> {
        let s = self.radial(p);
        let limit = if s > 1.0 {
            1.0 // hit the outside wall
        } else if s < Self::INNER {
            Self::INNER // hit the infield wall
        } else {
            return None;
        };
        Some(self.center + (p - self.center) / s * limit)
    }
}
