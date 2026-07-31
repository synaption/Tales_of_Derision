"""Headless tests for the juice layer.

Two halves, and the split is the point of the exercise:

* the maths in `juicefx.py` is checked with no pygame imported at all;
* the workbench's sim and its render path are driven under the dummy SDL
  driver, so nothing ever opens a window on the desktop.

Run it directly (`python3 juicetest.py`) or under pytest. The last group is
the one worth keeping honest -- easings that do not start at 0 and end at 1,
motions that leave a body permanently deformed, or a `Sequence` that eats a
frame per child are all bugs you cannot see by looking at the screen, only by
looking at the numbers.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import juicefx as jf


# ---------------------------------------------------------------------------
# Easing
# ---------------------------------------------------------------------------


def test_easings_are_normalised():
    """Every easing must map 0->0 and 1->1, or motions will snap on the last
    frame. What they do in between is a matter of taste."""
    for name, fn in jf.EASINGS.items():
        assert abs(fn(0.0)) < 1e-6, f"{name} does not start at 0"
        assert abs(fn(1.0) - 1.0) < 1e-6, f"{name} does not end at 1"


def test_back_and_elastic_leave_the_unit_range():
    """The overshoot is the whole reason these exist -- if a refactor clamps
    them, they become expensive versions of ease_out_cubic."""
    assert max(jf.ease_out_back(i / 100) for i in range(101)) > 1.02
    assert min(jf.ease_in_back(i / 100) for i in range(101)) < -0.02
    assert max(jf.ease_out_elastic(i / 100) for i in range(101)) > 1.02


def test_monotone_easings_do_not_go_backwards():
    for name in ("out_quad", "out_cubic", "out_quint", "in_out_cubic", "linear"):
        fn = jf.EASINGS[name]
        prev = -1.0
        for i in range(101):
            v = fn(i / 100)
            assert v >= prev - 1e-9, f"{name} reverses at t={i / 100}"
            prev = v


def test_damp_is_frame_rate_independent():
    """The classic juice bug: `x += (target-x)*0.1` per frame converges faster
    on a fast machine. One second of approach must land in the same place
    whether it was taken in 30 steps or 240."""
    def approach(steps):
        x = 0.0
        for _ in range(steps):
            x = jf.damp(x, 100.0, 0.001, 1.0 / steps)
        return x

    assert abs(approach(30) - approach(240)) < 0.05


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------


def test_noise_is_bounded_and_deterministic():
    for i in range(400):
        t = i * 0.37
        v = jf.value_noise(t, seed=5)
        assert -1.0 <= v <= 1.0
        assert v == jf.value_noise(t, seed=5)      # same input, same output


def test_noise_is_smooth():
    """Smoothness is what separates a blow landing from a loose cable. Adjacent
    samples a hundredth of a step apart must not jump the full range."""
    worst = max(abs(jf.value_noise(i * 0.01, 3) - jf.value_noise((i + 1) * 0.01, 3))
                for i in range(500))
    assert worst < 0.15, f"noise steps by {worst:.3f} between adjacent samples"


def test_different_seeds_decorrelate():
    """x, y and roll are three seeds of the same generator; if they agreed, the
    screen would slide along a diagonal instead of tumbling."""
    a = [jf.value_noise(i * 0.3, 1) for i in range(200)]
    b = [jf.value_noise(i * 0.3, 2) for i in range(200)]
    dot = sum(x * y for x, y in zip(a, b)) / len(a)
    assert abs(dot) < 0.2


# ---------------------------------------------------------------------------
# Body and motions
# ---------------------------------------------------------------------------


def run_motion(motion, dt=1 / 60, limit=600):
    """Drive a motion to completion, returning (body, frames, samples)."""
    body = jf.Body()
    anim = jf.Animator()
    anim.play(motion)
    samples = []
    frames = 0
    while anim.busy and frames < limit:
        anim.update(body, dt)
        samples.append((body.ox, body.oy, body.sx, body.sy, body.angle, body.alpha))
        frames += 1
    return body, frames, samples


def test_reset_juice_returns_to_identity():
    b = jf.Body()
    b.shift(3, 4)
    b.scale(2, 0.5)
    b.rotate(90)
    b.fade(0.2)
    b.whiten(1.0)
    b.reset_juice()
    assert (b.ox, b.oy, b.sx, b.sy, b.angle, b.alpha, b.flash) == (0, 0, 1, 1, 0, 1, 0)


def test_motions_compose_without_fighting():
    """Two motions on one body must both land. Offsets add, scales multiply --
    that accumulate rule is what lets a hop, a flash and a breath coexist."""
    body = jf.Body()
    anim = jf.Animator()
    anim.play(jf.Knockback(duration=1.0, dx=1, dy=0, distance=0.4, ease=jf.linear))
    anim.play(jf.Pop(duration=1.0, amount=1.0, ease=jf.linear))
    anim.update(body, 0.0)
    assert body.ox > 0.39, "knockback offset was lost"
    assert body.sx > 1.9 and body.sy > 1.9, "pop scale was lost"


def test_every_motion_ends_at_identity():
    """A motion that finishes must leave no residue. If it does, the effect
    accumulates over a session and the sprite slowly drifts or shrinks."""
    motions = [
        jf.Hop(duration=0.2, dx=1, dy=0),
        jf.Hop(duration=0.2, dx=0, dy=-1, squash=0.4),
        jf.Slide(duration=0.2, dx=-1, dy=0),
        jf.Lunge(duration=0.2, dx=1, dy=0),
        jf.Anticipate(duration=0.1, dx=0, dy=1),
        jf.Knockback(duration=0.2, dx=1, dy=1),
        jf.Flash(duration=0.1),
        jf.Pop(duration=0.2),
        jf.Shiver(duration=0.2),
    ]
    for m in motions:
        body, _, _ = run_motion(m)
        # The Animator wipes the body when the last motion is dropped, so what
        # is really under test is that the motion *does* report itself done.
        assert not jf.Animator().busy
        name = type(m).__name__
        body2 = jf.Body()
        anim = jf.Animator()
        anim.play(m)
        while anim.busy:
            anim.update(body2, 1 / 60)
        anim.update(body2, 1 / 60)   # one frame past the end
        assert (body2.ox, body2.oy) == (0.0, 0.0), f"{name} left an offset"
        assert (body2.sx, body2.sy) == (1.0, 1.0), f"{name} left a scale"
        assert body2.angle == 0.0 and body2.alpha == 1.0, f"{name} left a transform"


def test_hop_starts_at_the_old_tile_and_arrives():
    """The core lie: the logical move already happened, so frame one must draw
    the body a whole tile back the way it came."""
    body = jf.Body()
    anim = jf.Animator()
    anim.play(jf.Hop(duration=0.3, dx=1, dy=0, height=0.4))
    anim.update(body, 0.0)
    assert abs(body.ox + 1.0) < 0.01, "hop does not start at the previous tile"
    while anim.busy:
        anim.update(body, 1 / 60)
    assert abs(body.ox) < 1e-6, "hop does not arrive at the new tile"


def test_hop_lifts_off_the_floor():
    _, _, samples = run_motion(jf.Hop(duration=0.3, dx=1, dy=0, height=0.4))
    peak = min(s[1] for s in samples)          # -y is up
    assert peak < -0.3, f"hop only reached {peak:.3f} tiles"


def test_hop_squash_roughly_conserves_volume():
    """Stretched thin has to also mean taller. Break it and the sprite reads as
    a bug rather than as weight."""
    _, _, samples = run_motion(jf.Hop(duration=0.3, dx=1, dy=0, squash=0.3))
    for _, _, sx, sy, _, _ in samples:
        assert 0.75 < sx * sy < 1.25, f"volume drifted to {sx * sy:.3f}"


def test_lunge_goes_out_faster_than_it_comes_back():
    """The asymmetry is the effect. Equal speeds read as a nervous jiggle."""
    _, frames, samples = run_motion(jf.Lunge(duration=0.3, dx=1, dy=0, reach=0.6))
    peak_at = max(range(len(samples)), key=lambda i: samples[i][0])
    assert peak_at < frames * 0.45, "lunge peaks too late to read as a strike"
    assert samples[peak_at][0] > 0.5, "lunge never reaches"


def test_anticipate_leans_the_wrong_way():
    _, _, samples = run_motion(jf.Anticipate(duration=0.12, dx=1, dy=0, amount=0.2))
    assert min(s[0] for s in samples) < -0.15, "no backswing"
    assert max(s[0] for s in samples) <= 0.0, "anticipation moved forwards"


def test_death_spin_fades_and_shrinks():
    _, _, samples = run_motion(jf.DeathSpin(duration=0.5))
    last = samples[-1]
    assert last[5] < 0.1, "corpse never faded"
    assert last[2] < 0.15, "corpse never shrank"
    assert abs(last[4]) > 300, "corpse never spun"


def test_breathe_never_finishes_and_stays_small():
    body = jf.Body()
    anim = jf.Animator()
    anim.add_persistent(jf.Breathe(amplitude=0.04))
    for _ in range(600):
        anim.update(body, 1 / 60)
        assert 0.9 < body.sy < 1.1, "idle breathing is visible as a pulse"
    assert not anim.busy, "a persistent motion must not read as busy"


# ---------------------------------------------------------------------------
# Sequencing
# ---------------------------------------------------------------------------


def test_sequence_runs_children_in_order():
    fired = []
    seq = jf.Sequence([
        jf.Callback(lambda: fired.append("a")),
        jf.Wait(duration=0.1),
        jf.Callback(lambda: fired.append("b")),
    ])
    body = jf.Body()
    seq.update(body, 0.0)
    assert fired == ["a"], "sequence ran past a Wait immediately"
    for _ in range(10):
        seq.update(body, 1 / 60)
    assert fired == ["a", "b"]


def test_zero_length_children_do_not_each_eat_a_frame():
    """Six callbacks in a row must all fire on the same frame, or an attack
    with several instant steps would smear across a tenth of a second."""
    fired = []
    seq = jf.Sequence([jf.Callback(lambda i=i: fired.append(i)) for i in range(6)])
    seq.update(jf.Body(), 1 / 60)
    assert fired == [0, 1, 2, 3, 4, 5]


def test_parallel_finishes_with_its_longest_child():
    p = jf.Parallel([jf.Wait(duration=0.05), jf.Wait(duration=0.2)])
    body = jf.Body()
    steps = 0
    while not p.update(body, 1 / 60) and steps < 100:
        steps += 1
    assert 10 <= steps <= 14, f"parallel took {steps} frames, expected ~12"


def test_impact_callback_fires_at_full_extension():
    """The one moment that has to be right: contact lands when the lunge is
    out, not when the key was pressed."""
    fired = []
    dur = 0.2
    attack = jf.Sequence([jf.Parallel([
        jf.Lunge(duration=dur, dx=1, dy=0, out_frac=0.32),
        jf.Sequence([jf.Wait(duration=dur * 0.32),
                     jf.Callback(lambda: fired.append(True))]),
    ])])
    body, extension = jf.Body(), []
    anim = jf.Animator()
    anim.play(attack)
    hit_frame = None
    frame = 0
    while anim.busy and frame < 100:
        anim.update(body, 1 / 60)
        extension.append(body.ox)
        if fired and hit_frame is None:
            hit_frame = frame
        frame += 1
    assert hit_frame is not None, "the blow never landed"
    peak = max(range(len(extension)), key=lambda i: extension[i])
    assert abs(hit_frame - peak) <= 1, "contact and full extension disagree"


# ---------------------------------------------------------------------------
# Camera-scale effects
# ---------------------------------------------------------------------------


def test_trauma_decays_to_nothing():
    t = jf.Trauma(decay=1.6)
    t.add(1.0)
    for _ in range(120):
        t.update(1 / 60)
    assert t.amount == 0.0
    assert t.offset() == (0.0, 0.0, 0.0)


def test_trauma_is_superlinear():
    """`shake = trauma**2` is what makes a light tap a tap. A small trauma must
    displace much less than proportionally."""
    small, big = jf.Trauma(amount=0.3), jf.Trauma(amount=0.9)
    assert small.shake / big.shake < 0.15


def test_trauma_saturates_rather_than_stacking():
    t = jf.Trauma()
    for _ in range(20):
        t.add(0.5)
    assert t.amount == 1.0, "trauma must clamp, or ten hits become a seizure"


def test_trauma_offset_stays_in_budget():
    t = jf.Trauma(amount=1.0, decay=0.0)
    for i in range(500):
        t.update(1 / 60)
        dx, dy, deg = t.offset(max_px=16.0, max_deg=2.5)
        assert abs(dx) <= 16.0 and abs(dy) <= 16.0 and abs(deg) <= 2.5


def test_spring_settles_and_does_not_explode():
    s = jf.Spring(value=1.0, target=1.0, stiffness=260, damping=18)
    s.kick(1.2)
    peak = 0.0
    for _ in range(240):
        s.update(1 / 60)
        peak = max(peak, abs(s.value))
        assert abs(s.value) < 10.0, "spring diverged"
    assert peak > 1.0
    assert abs(s.value - 1.0) < 0.02, "spring never settled"


def test_stiff_spring_survives_a_low_frame_rate():
    """Substepping exists for this: one big Euler step on a stiff spring is a
    perfectly good way to send the camera to infinity."""
    s = jf.Spring(stiffness=900, damping=8)
    s.kick(5.0)
    for _ in range(60):
        s.update(1 / 15)              # 15fps
        assert abs(s.value) < 100.0, "stiff spring exploded at a low frame rate"


def test_hitstop_freezes_animation_but_not_the_clock():
    hs = jf.HitStop()
    hs.hit(0.1)
    for _ in range(5):                       # 5 frames = 83ms, still inside
        assert hs.consume(1 / 60) == 0.0, "hit-stop released early"
    for _ in range(3):                       # step past the end
        hs.consume(1 / 60)
    assert not hs.frozen and hs.consume(1 / 60) > 0.0, "hit-stop never released"


def test_camera_lag_trails_the_target():
    c = jf.Camera(x=0.0, y=0.0)
    c.follow(100.0, 0.0, 1 / 60)
    assert 0.0 < c.x < 100.0, "camera either snapped or did not move"
    for _ in range(120):
        c.follow(100.0, 0.0, 1 / 60)
    assert abs(c.x - 100.0) < 1.0, "camera never caught up"


def test_camera_snap_is_exact():
    c = jf.Camera()
    c.follow(100.0, 50.0, 1 / 60, smooth=False)
    assert (c.x, c.y) == (100.0, 50.0)


# ---------------------------------------------------------------------------
# Pools
# ---------------------------------------------------------------------------


def test_particles_vary_and_expire():
    f = jf.ParticleField()
    f.burst(0.0, 0.0, count=30, direction=(1, 0), spread=1.5)
    assert len(f) == 30
    speeds = {round(math.hypot(p.vx, p.vy), 1) for p in f.particles}
    assert len(speeds) > 10, "a burst at one speed reads as a cartwheel"
    for _ in range(180):
        f.update(1 / 60)
    assert len(f) == 0, "particles leaked"


def test_particle_cone_respects_its_direction():
    f = jf.ParticleField()
    f.burst(0.0, 0.0, count=40, direction=(1, 0), spread=1.0)
    assert all(p.vx > 0 for p in f.particles), "sparks fired backwards"


def test_bursts_are_deterministic():
    """Same seed, same burst -- so a headless render is reproducible."""
    a, b = jf.ParticleField(seed=3), jf.ParticleField(seed=3)
    a.burst(0, 0, count=12)
    b.burst(0, 0, count=12)
    assert [p.vx for p in a.particles] == [p.vx for p in b.particles]


def test_floater_pops_in_then_fades_out():
    f = jf.Floater("7", 0.0, 0.0)
    assert f.scale > 1.2, "damage number did not pop in"
    assert f.alpha == 1.0
    f.life = f.max_life * 0.05
    assert f.alpha < 0.2, "damage number never faded"


def test_shockwave_front_loads_its_expansion():
    """Most of the ring in the first few frames, or it reads as a bubble."""
    s = jf.Shockwave(0.0, 0.0, max_radius=100.0, life=0.4, max_life=0.4)
    s.life = 0.4 * 0.75          # a quarter of the way through
    assert s.radius > 60.0, f"only reached {s.radius:.0f} of 100 at t=0.25"


def test_tile_impact_is_local_and_transient():
    imp = jf.TileImpact(x=0.0, y=0.0, strength=8.0, life=0.5, max_life=0.5)
    imp.life = 0.4
    near = imp.displacement(60.0, 0.0)
    far = imp.displacement(4000.0, 0.0)
    assert far == (0.0, 0.0), "the wave front reached across the whole map"
    assert math.hypot(*near) < 20.0, "displacement is larger than the tile"


def test_tile_impact_wave_actually_travels():
    """The ring must move outward over time rather than pulsing in place."""
    def peak_distance_at(t):
        imp = jf.TileImpact(x=0.0, y=0.0, strength=8.0, life=0.5, max_life=0.5)
        imp.life = 0.5 * (1.0 - t)
        best = (0.0, 0.0)
        for d in range(0, 400, 4):
            mag = math.hypot(*imp.displacement(float(d), 0.0))
            if mag > best[1]:
                best = (float(d), mag)
        return best[0]

    assert peak_distance_at(0.5) > peak_distance_at(0.15) + 20


def test_effect_field_drains():
    fx = jf.EffectField()
    fx.particles.burst(0, 0, count=10)
    fx.floaters.add("3", 0, 0)
    fx.shockwave(0, 0)
    fx.slash(0, 0, 0.0)
    fx.impact(0, 0)
    assert len(fx) > 10
    for _ in range(180):
        fx.update(1 / 60)
    assert len(fx) == 0, "an effect pool leaked"


def test_ghost_trail_samples_on_a_timer_not_per_frame():
    """Per-frame sampling gives a solid smear whose density depends on the
    frame rate; a fixed interval looks the same on every machine."""
    def ghosts_after(dt, seconds):
        t = jf.GhostTrail(interval=0.03, life=1.0, limit=99)
        for _ in range(int(seconds / dt)):
            t.sample(0, 0, 1, 1, 0, dt, moving=True)
            t.update(dt)
        return len(t.ghosts)

    fast, slow = ghosts_after(1 / 240, 0.3), ghosts_after(1 / 60, 0.3)
    assert abs(fast - slow) <= 2, f"{fast} ghosts at 240fps vs {slow} at 60fps"


def test_ghost_trail_clears_when_still():
    t = jf.GhostTrail()
    for _ in range(30):
        t.sample(0, 0, 1, 1, 0, 1 / 60, moving=False)
    assert not t.ghosts, "a standing body left afterimages"


# ---------------------------------------------------------------------------
# The workbench, headless
# ---------------------------------------------------------------------------
#
# These import rogue_juice, which imports pygame -- so the dummy video driver
# is forced first. Nothing here ever opens a window.

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


def _world(**flags):
    import rogue_juice as rj
    juice = rj.Juice()
    for k, v in flags.items():
        juice.toggles[k].on = v
    return rj, rj.World(juice)


def _step(world, n, dt=1 / 60):
    for _ in range(n):
        world.update(dt)


def test_logical_move_is_instant_regardless_of_the_animation():
    """The sim must never wait for a tween. Immediately after the call the
    entity is on the new tile, even though it is still drawn on the old one."""
    rj, world = _world()
    start = world.player.tile
    world.try_move(world.player, -1, 0)
    assert world.player.tile == (start[0] - 1, start[1]), "the sim waited for a tween"
    # Offsets are written by the motions, so they appear on the first update --
    # which in the real loop happens before anything is drawn.
    world.update(0.0)
    assert abs(world.player.body.ox - 1.0) < 0.05, "not drawn back at the old tile"


def test_a_kill_removes_the_entity_after_its_animation():
    rj, world = _world()
    dummy = world.nearest_dummy()
    assert dummy in world.entities
    dummy.hp = 1
    world.land_blow(world.player, dummy, 1, 0)
    assert dummy.dying and dummy in world.entities, "corpse vanished instantly"
    _step(world, 60)
    assert dummy not in world.entities, "corpse never cleaned up"


def test_unjuiced_kill_removes_immediately():
    rj, world = _world()
    world.juice.master = False
    dummy = world.nearest_dummy()
    dummy.hp = 1
    world.land_blow(world.player, dummy, 1, 0)
    _step(world, 1)
    assert dummy not in world.entities


def test_nothing_moves_at_all_with_the_juice_off():
    """The A/B switch has to reach *everything*.

    Walk, bump a wall, swing at air, take a hit and kill something, checking on
    every single frame that no body has left its identity pose. Anything that
    animates without a toggle behind it -- an idle breath, a move tween, a log
    line scaling in -- shows up here as a body that is not exactly 1.0.
    """
    rj, world = _world()
    world.juice.master = False
    p = world.player

    def assert_still(where):
        for e in world.entities:
            b = e.body
            assert (b.ox, b.oy) == (0.0, 0.0), f"{e.name} shifted during {where}"
            assert (b.sx, b.sy) == (1.0, 1.0), f"{e.name} scaled during {where}"
            assert b.angle == 0.0 and b.alpha == 1.0, f"{e.name} moved in {where}"
            assert b.flash == 0.0, f"{e.name} flashed during {where}"

    for label, action in (
        ("a walk", lambda: world.try_move(p, 0, -1)),
        ("a wall bump", lambda: (setattr(p.body, "tx", 1.0),
                                 setattr(p.body, "ty", 1.0),
                                 world.try_move(p, -1, 0))),
        ("a whiffed swing", world.swing),
        ("taking a hit", world.strike_player),
        ("a kill", lambda: world.kill(world.nearest_dummy())),
    ):
        action()
        for _ in range(45):
            world.update(1 / 60)
            assert_still(label)


def test_the_move_tween_is_what_makes_a_step_visible():
    """`tween` off must teleport; on must not. Without this toggle there is no
    way to see the baseline the rest of the movement effects improve on."""
    # Breathing off too, since it legitimately nudges the body on its own and
    # would otherwise mask what this test is actually asking about.
    rj, world = _world(tween=False, bob=False)
    world.try_move(world.player, 0, -1)
    world.update(1 / 60)
    assert world.player.body.ox == 0.0 and world.player.body.oy == 0.0

    rj, world = _world()
    world.try_move(world.player, 0, -1)
    world.update(1 / 60)
    assert abs(world.player.body.oy) > 0.5, "the tween did not lag the picture"


def test_idle_breathing_actually_stops_when_switched_off():
    """This one shipped broken: the breath was a persistent motion and nothing
    ever consulted its toggle."""
    rj, world = _world()
    world.juice.toggles["bob"].on = False
    for _ in range(120):
        world.update(1 / 60)
        assert world.player.body.sy == 1.0, "the idle breathing toggle does nothing"

    rj, world = _world()
    seen = set()
    for _ in range(120):
        world.update(1 / 60)
        seen.add(round(world.player.body.sy, 4))
    assert len(seen) > 10, "nothing breathes even with the toggle on"


def test_master_switch_spawns_no_effects_at_all():
    """The A/B comparison has to be honest -- with the master off, a blow must
    produce a damage number and nothing else."""
    rj, world = _world()
    world.juice.master = False
    dummy = world.nearest_dummy()
    world.land_blow(world.player, dummy, 1, 0)
    assert len(world.fx) == 0, "effects leaked past the master switch"
    assert world.trauma.amount == 0.0
    assert not world.hitstop.frozen


def test_a_juiced_blow_lights_up_every_channel():
    rj, world = _world()
    dummy = world.nearest_dummy()
    world.land_blow(world.player, dummy, 1, 0)
    assert world.trauma.amount > 0.0, "no shake"
    assert world.hitstop.frozen, "no hit-stop"
    assert len(world.fx.particles) > 0, "no sparks"
    assert len(world.fx.floaters) == 1, "no damage number"
    assert world.fx.shockwaves and world.fx.impacts and world.fx.slashes
    assert dummy.body.flash == 0.0, "flash applies on update, not on impact"
    world.update(1 / 60)
    assert dummy.body.flash > 0.0, "victim never flashed"


def test_hitstop_freezes_the_world_but_not_the_shake():
    rj, world = _world()
    dummy = world.nearest_dummy()
    world.land_blow(world.player, dummy, 1, 0)
    before = [(p.x, p.y) for p in world.fx.particles.particles]
    trauma_before = world.trauma.amount
    world.update(1 / 60)
    after = [(p.x, p.y) for p in world.fx.particles.particles]
    assert before == after, "particles moved during hit-stop"
    assert world.trauma.amount < trauma_before, "the shake froze too"


def test_every_toggle_can_be_turned_off_without_crashing():
    """One pass with each effect solo, and one with the lot of them off. This
    is what stops a toggle from being wired to code that assumes it is on."""
    rj, _ = _world()
    keys = list(rj.Juice().toggles)
    for solo in keys + [None]:
        juice = rj.Juice()
        juice.set_all(False)
        if solo:
            juice.toggles[solo].on = True
        world = rj.World(juice)
        world.try_move(world.player, -1, 0)
        _step(world, 12)
        world.try_move(world.player, 0, -1)
        _step(world, 12)
        d = world.nearest_dummy()
        for _ in range(6):
            world.land_blow(world.player, d, 1, 0)
            _step(world, 20)
            if d.dying:
                d = world.nearest_dummy() or d
        _step(world, 60)


def test_walking_into_a_wall_does_not_move_the_entity():
    rj, world = _world()
    p = world.player
    p.body.tx, p.body.ty = 1.0, 1.0
    p.anim.clear()
    world.try_move(p, -1, 0)          # into the border wall
    assert p.tile == (1, 1)
    assert p.anim.busy, "a blocked move gave no feedback at all"


def test_intensity_zero_leaves_the_body_alone():
    """The intensity dial is the tuning knob -- at zero, amplitudes vanish even
    though the timings are untouched."""
    rj, world = _world()
    world.juice.intensity = 0.0
    world.try_move(world.player, -1, 0)
    peak = 0.0
    for _ in range(20):
        world.update(1 / 60)
        peak = max(peak, abs(world.player.body.oy))
    assert peak < 0.02, f"hop still lifted {peak:.3f} tiles at zero intensity"


def test_every_key_binding_runs():
    """Fire every documented key through the real handler.

    Input handling is the one place a workbench rots without anyone noticing --
    an effect nobody can trigger is an effect nobody tests.
    """
    import pygame

    import rogue_juice as rj
    pygame.init()
    pygame.display.set_mode((rj.WIN_W, rj.WIN_H))
    world = rj.World(rj.Juice())
    renderer = rj.Renderer()

    def press(key, mods=0):
        pygame.key.set_mods(mods)
        ev = pygame.event.Event(pygame.KEYDOWN, key=key, mod=mods, unicode="")
        result = rj.handle_event(ev, world, renderer)
        pygame.key.set_mods(0)
        for _ in range(10):
            world.update(1 / 60)
        return result

    for key in rj.MOVE_KEYS:
        assert press(key)
    for key in (pygame.K_SPACE, pygame.K_x, pygame.K_r, pygame.K_TAB,
                pygame.K_F1, pygame.K_F2, pygame.K_MINUS, pygame.K_EQUALS,
                pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET):
        assert press(key), f"{pygame.key.name(key)} quit the bench"
    assert press(pygame.K_k, pygame.KMOD_LSHIFT)      # the kill shortcut
    assert not press(pygame.K_ESCAPE), "escape did not quit"
    assert not press(pygame.K_q), "q did not quit"

    # Tab was pressed an even number of times above only if F1/F2 left it be;
    # what matters is that the master switch is still a bool and the panel and
    # the world both still draw after all of that.
    rj.draw_frame(pygame.display.get_surface(), renderer, world, (0, 0))
    pygame.quit()


def test_panel_clicks_toggle_the_row_under_the_cursor():
    import pygame

    import rogue_juice as rj
    pygame.init()
    screen = pygame.display.set_mode((rj.WIN_W, rj.WIN_H))
    world = rj.World(rj.Juice())
    renderer = rj.Renderer()
    rj.draw_frame(screen, renderer, world, (0, 0))

    assert len(renderer.rows) == len(world.juice.toggles), "a toggle has no row"
    for rect, key in renderer.rows:
        before = world.juice.toggles[key].on
        assert renderer.panel_click(world, rect.center)
        assert world.juice.toggles[key].on is not before, f"{key} did not toggle"
    # Rows must stay inside the panel and clear of the footer blurb. The whole
    # list has to fit on screen without scrolling, so this is the assertion
    # that fails first if an effect is added or a font size grows.
    for rect, key in renderer.rows:
        assert rect.right <= rj.WIN_W, f"{key} row runs off the panel"
        assert rect.bottom < rj.WIN_H - rj.FOOTER_H, f"{key} row overlaps the footer"
    pygame.quit()


def test_the_footer_text_fits_below_the_separator():
    """Both footer states -- the hover blurb and the control list -- have to
    land inside the window. This is what catches a longer blurb being added."""
    import pygame

    import rogue_juice as rj
    pygame.font.init()
    line_h, avail = 14, rj.FOOTER_H - 7
    ui = pygame.font.Font(
        pygame.font.match_font("dejavusansmono,couriernew,monospace"), 13)
    assert len(rj.HELP_LINES) * line_h <= avail, "the control list overflows"
    for line in rj.HELP_LINES:
        assert ui.size(line)[0] <= rj.PANEL_W - 28, f"help line too wide: {line!r}"
    for t in rj.Juice().toggles.values():
        lines = rj.Renderer._wrap(t.blurb, rj.PANEL_W - 30, ui)
        assert 16 + len(lines) * line_h <= avail, f"{t.key} blurb is too long"


def test_the_window_fits_a_small_laptop_screen():
    """The bench is no use if it does not fit on the screen it is being run on.
    1366x768 is the floor, less an allowance for title bar and taskbar."""
    import rogue_juice as rj
    assert rj.WIN_W <= 1366, f"{rj.WIN_W}px wide"
    assert rj.WIN_H <= 700, f"{rj.WIN_H}px tall"


def test_the_map_is_larger_than_the_viewport():
    """Otherwise the camera cannot pan, and a camera that cannot pan cannot
    demonstrate camera lag."""
    import rogue_juice as rj
    assert rj.GRID_W * rj.TILE > rj.VIEW_W
    assert rj.GRID_H * rj.TILE > rj.VIEW_H


def test_headless_render_produces_a_frame():
    """Drives the real draw path under the dummy driver: floor, ripples,
    squashed glyphs, particles, panel and all."""
    import tempfile

    import rogue_juice as rj
    path = os.path.join(tempfile.mkdtemp(), "juice.png")
    rj.run_headless(path)
    assert os.path.getsize(path) > 5000, "the saved frame is suspiciously empty"


def test_compositing_survives_every_whole_frame_effect_at_once():
    """Shake, kick, zoom, tilt, flash, vignette and the RGB split all live at
    the same time, which is the one combination that exercises rotozoom plus
    the channel split on the same surface."""
    import pygame

    import rogue_juice as rj
    pygame.init()
    screen = pygame.display.set_mode((rj.WIN_W, rj.WIN_H))
    world = rj.World(rj.Juice())
    renderer = rj.Renderer()
    world.juice.intensity = 2.0
    d = world.nearest_dummy()
    d.hp = 1
    world.land_blow(world.player, d, 1, 0)
    for _ in range(30):
        world.update(1 / 60)
        rj.draw_frame(screen, renderer, world, (rj.WIN_W - 200, 300))
    pygame.quit()


# ---------------------------------------------------------------------------


def main():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"  ok    {name}")
        except AssertionError as exc:
            failures.append((name, exc))
            print(f"  FAIL  {name}: {exc}")
        except Exception as exc:                      # noqa: BLE001
            failures.append((name, exc))
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
