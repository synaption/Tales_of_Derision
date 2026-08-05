//! A small deterministic random number generator.
//!
//! Deliberately not the `rand` crate. The city, the squad and every ant wave
//! are generated from one seed, and the tests assert that two `Sim`s built
//! from the same seed are identical -- which means the generator has to be
//! part of the program rather than a dependency whose stream can change
//! between minor versions or between platforms.
//!
//! PCG-XSH-RR 32/64: one 64-bit LCG state, output permuted by an
//! xorshift-and-rotate. It is nine lines, passes the statistical tests that
//! matter here, and gives the same sequence everywhere.

use bevy_ecs::resource::Resource;

/// The multiplier from the reference PCG implementation.
const MULTIPLIER: u64 = 6_364_136_223_846_793_005;

/// A seeded stream of pseudo-random numbers.
#[derive(Debug, Clone, Resource)]
pub struct Rng {
    state: u64,
    /// Stream selector. Always odd, which is what keeps two streams from
    /// ever converging onto the same sequence.
    increment: u64,
}

impl Rng {
    /// Start a stream from a seed. Two `Rng`s with the same seed agree
    /// forever; two with different seeds do not correlate.
    pub fn new(seed: u64) -> Self {
        let mut rng = Rng {
            state: 0,
            increment: (seed << 1) | 1,
        };
        rng.next_u32();
        rng.state = rng.state.wrapping_add(seed);
        rng.next_u32();
        rng
    }

    /// The raw generator: advance the LCG, then permute the *old* state.
    ///
    /// Permuting the previous state rather than the new one is what makes
    /// this PCG rather than a plain LCG, and it is the reason the low bits
    /// are usable -- an LCG's low bits have a period of 2, which is exactly
    /// the failure mode that bites anyone who writes `state % n`.
    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(MULTIPLIER).wrapping_add(self.increment);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rotation = (old >> 59) as u32;
        xorshifted.rotate_right(rotation)
    }

    /// A float in `[0, 1)`.
    ///
    /// Uses the top 24 bits, which is exactly the mantissa of an `f32`, so
    /// every value is representable and no rounding pushes a result to 1.0.
    pub fn unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }

    /// A float in `[low, high)`.
    pub fn range(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * self.unit()
    }

    /// An integer in `[0, bound)`, or 0 if `bound` is 0.
    ///
    /// Debiased by rejection: `next_u32() % bound` favours the low end
    /// whenever `bound` does not divide 2^32, and the wave spawner picks
    /// enough numbers for that to be visible as a lopsided city.
    pub fn below(&mut self, bound: u32) -> u32 {
        if bound == 0 {
            return 0;
        }
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let value = self.next_u32();
            if value >= threshold {
                return value % bound;
            }
        }
    }

    /// An integer in `[low, high]`, inclusive at both ends.
    pub fn between(&mut self, low: i32, high: i32) -> i32 {
        if high <= low {
            return low;
        }
        low + self.below((high - low + 1) as u32) as i32
    }

    /// True with probability `chance`.
    pub fn chance(&mut self, chance: f32) -> bool {
        self.unit() < chance
    }
}

#[cfg(test)]
mod tests {
    use super::Rng;

    #[test]
    fn same_seed_gives_the_same_stream() {
        let mut a = Rng::new(1234);
        let mut b = Rng::new(1234);
        for _ in 0..1000 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }

    #[test]
    fn different_seeds_diverge() {
        let mut a = Rng::new(1234);
        let mut b = Rng::new(1235);
        let differ = (0..64).filter(|_| a.next_u32() != b.next_u32()).count();
        assert!(differ > 60, "streams tracked each other: {differ}/64 differed");
    }

    #[test]
    fn unit_stays_in_range_and_spreads_out() {
        let mut rng = Rng::new(7);
        let mut buckets = [0u32; 10];
        for _ in 0..10_000 {
            let value = rng.unit();
            assert!((0.0..1.0).contains(&value), "out of range: {value}");
            buckets[(value * 10.0) as usize] += 1;
        }
        // A fair generator puts 1000 in each. Anything outside 800-1200 is
        // a real skew rather than noise at this sample count.
        for (index, count) in buckets.iter().enumerate() {
            assert!(
                (800..1200).contains(count),
                "decile {index} held {count} of 10000"
            );
        }
    }

    #[test]
    fn below_is_unbiased_at_an_awkward_bound() {
        // 3 does not divide 2^32, so a naive modulo skews toward 0 and 1.
        let mut rng = Rng::new(99);
        let mut counts = [0u32; 3];
        for _ in 0..30_000 {
            counts[rng.below(3) as usize] += 1;
        }
        for (value, count) in counts.iter().enumerate() {
            assert!(
                (9_500..10_500).contains(count),
                "value {value} came up {count} times in 30000"
            );
        }
    }
}
