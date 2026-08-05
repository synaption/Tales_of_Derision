//! Uniform-grid spatial indexes, for "what is near me" without the N squared.
//!
//! Two of them, because the two questions have different shapes:
//!
//! * [`SpatialHash`] is rebuilt every frame from moving points -- the ants,
//!   the allies, the player. Target selection and projectile hits both ask it.
//! * [`StaticGrid`] is built once from the city's boxes and then only read.
//!   Collision asks it, sixty times a second, for every moving body.
//!
//! A city block is about 20m and a bug is about 2m, so a 12m cell keeps both
//! queries down to a handful of buckets.
//!
//! Neither structure allocates during a query. The callers hand in the
//! `Vec` to fill, which in a Bevy system is a `Local<Vec<_>>` that keeps its
//! capacity between frames -- and that matters, because the answer is
//! "nothing" a few thousand times a second and allocating to say so is the
//! kind of cost that only shows up as a mysteriously uneven frame time.
//!
//! Only the horizontal plane is indexed. Altitude is not: a Wing Diver 60m up
//! is still "at" the intersection she is above, and every consumer wants that
//! -- ants track the ground position of a flyer and merely fail to reach her,
//! which is the correct behaviour.

use bevy_ecs::prelude::*;
use bevy_platform::collections::{HashMap, HashSet};

/// Floor-divide a world coordinate into a cell index.
///
/// `as i32` truncates toward zero, which would fold the cells either side of
/// the origin into one and make a two-metre seam down the middle of the map
/// where collisions are missed. `floor` first.
#[inline]
fn cell_of(value: f32, cell: f32) -> i32 {
    (value / cell).floor() as i32
}

// --------------------------------------------------------------------------
// moving points
// --------------------------------------------------------------------------

/// A rebuilt-every-frame bucket grid over 2D points.
///
/// Rebuilding from scratch beats maintaining it incrementally at this entity
/// count -- a few hundred inserts is cheaper than the bookkeeping needed to
/// keep a dirty-list correct, and it cannot go stale.
#[derive(Resource, Debug, Default)]
pub struct SpatialHash {
    cell: f32,
    buckets: HashMap<(i32, i32), Vec<(Entity, f32, f32)>>,
}

impl SpatialHash {
    pub fn new(cell: f32) -> Self {
        SpatialHash {
            cell,
            buckets: HashMap::default(),
        }
    }

    /// Empty every bucket but keep their allocations for this frame's refill.
    pub fn clear(&mut self) {
        for bucket in self.buckets.values_mut() {
            bucket.clear();
        }
    }

    pub fn insert(&mut self, entity: Entity, x: f32, z: f32) {
        let key = (cell_of(x, self.cell), cell_of(z, self.cell));
        self.buckets.entry(key).or_default().push((entity, x, z));
    }

    /// Append `(entity, distance_squared)` for every point within `radius`.
    ///
    /// Unordered, and does not clear `out` -- callers that want the nearest
    /// should take the minimum rather than sorting, because they almost
    /// always only want one.
    pub fn query_radius(&self, x: f32, z: f32, radius: f32, out: &mut Vec<(Entity, f32)>) {
        let r2 = radius * radius;
        let (cx0, cz0) = (cell_of(x - radius, self.cell), cell_of(z - radius, self.cell));
        let (cx1, cz1) = (cell_of(x + radius, self.cell), cell_of(z + radius, self.cell));
        for cx in cx0..=cx1 {
            for cz in cz0..=cz1 {
                let Some(bucket) = self.buckets.get(&(cx, cz)) else {
                    continue;
                };
                for &(entity, px, pz) in bucket {
                    let (dx, dz) = (px - x, pz - z);
                    let d2 = dx * dx + dz * dz;
                    if d2 <= r2 {
                        out.push((entity, d2));
                    }
                }
            }
        }
    }

    /// Closest point within `radius` that is in `accept`, and its distance
    /// squared. `None` if there is nothing.
    ///
    /// Filtering during the walk rather than after is the point of taking
    /// `accept`: pass the set of enemy entities and the search never even
    /// measures a friend.
    pub fn nearest(
        &self,
        x: f32,
        z: f32,
        radius: f32,
        accept: &HashSet<Entity>,
    ) -> Option<(Entity, f32)> {
        let r2 = radius * radius;
        let (cx0, cz0) = (cell_of(x - radius, self.cell), cell_of(z - radius, self.cell));
        let (cx1, cz1) = (cell_of(x + radius, self.cell), cell_of(z + radius, self.cell));
        let mut best: Option<(Entity, f32)> = None;
        for cx in cx0..=cx1 {
            for cz in cz0..=cz1 {
                let Some(bucket) = self.buckets.get(&(cx, cz)) else {
                    continue;
                };
                for &(entity, px, pz) in bucket {
                    let (dx, dz) = (px - x, pz - z);
                    let d2 = dx * dx + dz * dz;
                    if d2 > r2 || !accept.contains(&entity) {
                        continue;
                    }
                    if best.is_none_or(|(_, b)| d2 < b) {
                        best = Some((entity, d2));
                    }
                }
            }
        }
        best
    }
}

// --------------------------------------------------------------------------
// static boxes
// --------------------------------------------------------------------------

/// One building's footprint: an axis-aligned rectangle on the ground plane
/// and the height of its roof. Every box starts at `y = 0` -- the city has no
/// overhangs, and collision is much cheaper for it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Slab {
    pub x0: f32,
    pub z0: f32,
    pub x1: f32,
    pub z1: f32,
    pub top: f32,
}

impl Slab {
    /// Is `(x, z)` inside the footprint?
    #[inline]
    pub fn contains(&self, x: f32, z: f32) -> bool {
        x >= self.x0 && x <= self.x1 && z >= self.z0 && z <= self.z1
    }
}

/// Build-once index of axis-aligned boxes, keyed by the cells they cover.
#[derive(Resource, Debug, Default)]
pub struct StaticGrid {
    cell: f32,
    buckets: HashMap<(i32, i32), Vec<u32>>,
    slabs: Vec<Slab>,
}

impl StaticGrid {
    pub fn new(cell: f32) -> Self {
        StaticGrid {
            cell,
            buckets: HashMap::default(),
            slabs: Vec::new(),
        }
    }

    pub fn slabs(&self) -> &[Slab] {
        &self.slabs
    }

    pub fn add(&mut self, slab: Slab) {
        let index = self.slabs.len() as u32;
        self.slabs.push(slab);
        for cx in cell_of(slab.x0, self.cell)..=cell_of(slab.x1, self.cell) {
            for cz in cell_of(slab.z0, self.cell)..=cell_of(slab.z1, self.cell) {
                self.buckets.entry((cx, cz)).or_default().push(index);
            }
        }
    }

    /// Fill `out` with every stored box whose cells overlap the query
    /// rectangle.
    ///
    /// Cell overlap is a superset of box overlap, so callers still have to
    /// test properly -- they were going to anyway, to find the penetration
    /// depth.
    ///
    /// A box straddling several cells is found several times, so the results
    /// are de-duplicated by scanning what has already been pushed. That is
    /// quadratic, and it is still the right call: the answer is one or two
    /// buildings, and a linear scan of two elements beats hashing them.
    pub fn query_aabb(&self, x0: f32, z0: f32, x1: f32, z1: f32, out: &mut Vec<Slab>) {
        out.clear();
        for cx in cell_of(x0, self.cell)..=cell_of(x1, self.cell) {
            for cz in cell_of(z0, self.cell)..=cell_of(z1, self.cell) {
                let Some(bucket) = self.buckets.get(&(cx, cz)) else {
                    continue;
                };
                for &index in bucket {
                    let slab = self.slabs[index as usize];
                    if !out.contains(&slab) {
                        out.push(slab);
                    }
                }
            }
        }
    }

    /// Tallest roof directly over `(x, z)`, or 0 for open street.
    ///
    /// Does its own bucket walk rather than going through
    /// [`StaticGrid::query_aabb`] so that it needs no scratch buffer: it is
    /// called from spawn code and from the camera, neither of which is in a
    /// position to be holding one, and a duplicate box cannot change a
    /// maximum anyway.
    pub fn height_at(&self, x: f32, z: f32) -> f32 {
        let key = (cell_of(x, self.cell), cell_of(z, self.cell));
        let Some(bucket) = self.buckets.get(&key) else {
            return 0.0;
        };
        let mut best = 0.0f32;
        for &index in bucket {
            let slab = self.slabs[index as usize];
            if slab.contains(x, z) && slab.top > best {
                best = slab.top;
            }
        }
        best
    }

    /// Is the point inside any building, treating each box as solid from the
    /// ground to its roof? Used by projectiles and by the camera boom.
    pub fn contains_point(&self, x: f32, y: f32, z: f32) -> bool {
        let key = (cell_of(x, self.cell), cell_of(z, self.cell));
        let Some(bucket) = self.buckets.get(&key) else {
            return false;
        };
        bucket
            .iter()
            .any(|&index| {
                let slab = self.slabs[index as usize];
                slab.contains(x, z) && y <= slab.top
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid() -> StaticGrid {
        let mut grid = StaticGrid::new(16.0);
        grid.add(Slab { x0: -5.0, z0: -5.0, x1: 5.0, z1: 5.0, top: 20.0 });
        grid.add(Slab { x0: 40.0, z0: 40.0, x1: 60.0, z1: 60.0, top: 8.0 });
        grid
    }

    #[test]
    fn cells_do_not_fold_across_the_origin() {
        // -1.0 and +1.0 with a 16m cell must land in cells -1 and 0. A
        // truncating cast puts both in cell 0, which silently drops every
        // collision in the quadrant behind the player's spawn.
        assert_eq!(cell_of(-1.0, 16.0), -1);
        assert_eq!(cell_of(1.0, 16.0), 0);
        assert_eq!(cell_of(-16.0, 16.0), -1);
        assert_eq!(cell_of(-17.0, 16.0), -2);
    }

    #[test]
    fn a_box_spanning_cells_is_returned_once() {
        let mut grid = StaticGrid::new(4.0);
        // 20m across a 4m grid: this box lives in twenty-five cells.
        grid.add(Slab { x0: -10.0, z0: -10.0, x1: 10.0, z1: 10.0, top: 5.0 });
        let mut out = Vec::new();
        grid.query_aabb(-10.0, -10.0, 10.0, 10.0, &mut out);
        assert_eq!(out.len(), 1, "duplicates leaked: {out:?}");
    }

    #[test]
    fn height_at_finds_roofs_and_street() {
        let grid = grid();
        assert_eq!(grid.height_at(0.0, 0.0), 20.0);
        assert_eq!(grid.height_at(50.0, 50.0), 8.0);
        assert_eq!(grid.height_at(25.0, 0.0), 0.0, "open street is zero");
    }

    #[test]
    fn contains_point_is_solid_from_ground_to_roof() {
        let grid = grid();
        assert!(grid.contains_point(0.0, 0.5, 0.0));
        assert!(grid.contains_point(0.0, 19.9, 0.0));
        assert!(!grid.contains_point(0.0, 20.1, 0.0), "above the roof is sky");
        assert!(!grid.contains_point(30.0, 1.0, 0.0), "the street is not solid");
    }

    #[test]
    fn query_aabb_returns_only_overlapping_cells() {
        let grid = grid();
        let mut out = Vec::new();
        grid.query_aabb(-1.0, -1.0, 1.0, 1.0, &mut out);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].top, 20.0);

        grid.query_aabb(200.0, 200.0, 201.0, 201.0, &mut out);
        assert!(out.is_empty(), "empty query must clear the buffer");
    }

    #[test]
    fn nearest_filters_by_the_accept_set() {
        let mut world = World::new();
        let friend = world.spawn_empty().id();
        let foe = world.spawn_empty().id();

        let mut hash = SpatialHash::new(12.0);
        hash.insert(friend, 1.0, 0.0);
        hash.insert(foe, 9.0, 0.0);

        let mut enemies = HashSet::default();
        enemies.insert(foe);

        let (found, d2) = hash.nearest(0.0, 0.0, 50.0, &enemies).expect("foe in range");
        assert_eq!(found, foe, "the closer entity was not in the accept set");
        assert!((d2 - 81.0).abs() < 1e-3);

        assert!(
            hash.nearest(0.0, 0.0, 5.0, &enemies).is_none(),
            "the foe is 9m away and the radius was 5m"
        );
    }

    #[test]
    fn clearing_keeps_the_hash_usable() {
        let mut world = World::new();
        let entity = world.spawn_empty().id();
        let mut hash = SpatialHash::new(12.0);
        hash.insert(entity, 0.0, 0.0);
        hash.clear();

        let mut out = Vec::new();
        hash.query_radius(0.0, 0.0, 100.0, &mut out);
        assert!(out.is_empty());

        hash.insert(entity, 3.0, 4.0);
        hash.query_radius(0.0, 0.0, 100.0, &mut out);
        assert_eq!(out.len(), 1);
        assert!((out[0].1 - 25.0).abs() < 1e-3);
    }
}
