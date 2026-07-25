"""Gameplay interaction logic: targeting, feature use (chop/drink/cook/harvest),
crafting/building, trade/loot, social reads, and look-mode queries.

Renderer-free by design -- every function here takes plain data / entity ids and
returns values or messages, never a ``Renderer`` -- so it is unit-testable headless
and the UI layer (``ui`` / the turn loop in ``main``) drives it. Split out of
``main`` so game flow and presentation stay separate from this logic.
"""
from __future__ import annotations

from dataclasses import dataclass

import esper

from components import (
    Bed, Blueprint, Corpse, Deer, Dialogue, Enemy, Equipment, Friendly, Inventory,
    NPC, Name, Needs, Personality, Player, Position, Relationships, Renderable, Tree,
)
from game_map import GameMap
from items import (
    BERRIES, WOOD, WOOD_DOOR, WOOD_WALL, WOOD_WINDOW, cook_meat, craft_cost,
    hunger_restored, is_raw_meat, placed_tile, thirst_restored,
)
from queries import entity_name, first_player_entity
from systems import (
    MovementProcessor, WAIT_ACTION, active_statuses, friendship, interact,
    pick_berries, queue_message, raise_blueprint, slay_entity, status_label,
    stock_blueprint, world_clock,
)


_CARDINAL_ACTION_DELTAS = {
    "move_up": (0, -1),
    "move_down": (0, 1),
    "move_left": (-1, 0),
    "move_right": (1, 0),
}


_VECTOR_TO_ACTION = {
    (-1, -1): "move_up_left",
    (0, -1): "move_up",
    (1, -1): "move_up_right",
    (-1, 0): "move_left",
    (1, 0): "move_right",
    (-1, 1): "move_down_left",
    (0, 1): "move_down",
    (1, 1): "move_down_right",
}


def _infer_slot_for_item(item_name: str) -> str | None:
    lowered = item_name.lower()
    if any(token in lowered for token in ("helm", "hood", "hat", "crown")):
        return "head"
    if any(token in lowered for token in ("chest", "tunic", "armor", "robe", "coat")):
        return "chest"
    if any(token in lowered for token in ("glove", "gauntlet")):
        return "hands"
    if any(token in lowered for token in ("pants", "greave", "leggings", "trousers")):
        return "legs"
    if any(token in lowered for token in ("boot", "shoe", "sandal")):
        return "feet"
    if any(token in lowered for token in ("shield", "buckler", "offhand", "off-hand")):
        return "off hand"
    if any(token in lowered for token in ("ring",)):
        return "ring"
    if any(token in lowered for token in ("sword", "dagger", "axe", "mace", "club", "staff", "spear", "bow")):
        return "main hand"
    return None


def _equip_inventory_item(inventory: Inventory, equipment: Equipment, item_index: int) -> str:
    if item_index < 0 or item_index >= len(inventory.items):
        return "No item selected."

    item_name = inventory.items[item_index]
    slot_name = _infer_slot_for_item(item_name)
    if slot_name is None:
        return f"Cannot equip {item_name}."

    if slot_name not in equipment.slots:
        equipment.slots[slot_name] = None

    replaced = equipment.slots.get(slot_name)
    equipment.slots[slot_name] = item_name
    inventory.items.pop(item_index)
    if replaced:
        inventory.items.append(replaced)
        return f"Equipped {item_name} to {slot_name}; unequipped {replaced}."
    return f"Equipped {item_name} to {slot_name}."


def _unequip_slot(inventory: Inventory, equipment: Equipment, slot_name: str) -> str:
    current = equipment.slots.get(slot_name)
    if not current:
        return f"Nothing equipped in {slot_name}."

    equipment.slots[slot_name] = None
    inventory.items.append(current)
    return f"Unequipped {current} from {slot_name}."


def _item_visual(item_name: str) -> tuple[str, str, tuple[int, int, int] | None, tuple[int, int, int] | None]:
    lowered = item_name.lower()

    if any(token in lowered for token in ("sword", "dagger", "knife", "rapier", "spear", "javelin", "halberd", "axe", "mace", "club", "staff", "bow", "crossbow")):
        return (")", "valuable", (220, 220, 220), None)

    if any(token in lowered for token in ("helm", "hood", "hat", "crown", "chest", "tunic", "armor", "robe", "coat", "shield", "buckler", "pants", "greave", "leggings", "trousers", "boot", "shoe", "sandal", "glove", "gauntlet")):
        return ("[", "valuable", (176, 188, 214), None)

    if any(token in lowered for token in ("potion", "flask", "waterskin")):
        return ("!", "valuable", (120, 196, 255), None)

    if any(token in lowered for token in ("bandage", "scroll", "map", "book", "torch")):
        return ("?", "valuable", (236, 210, 150), None)

    if any(token in lowered for token in ("wood", "log", "branch", "kindling")):
        return ("=", "valuable", (150, 111, 70), None)

    if "cooked" in lowered and "meat" in lowered:
        return ("%", "valuable", (196, 118, 74), None)

    if any(token in lowered for token in ("apple", "bread", "meat", "food")):
        return ("%", "valuable", (222, 142, 106), None)

    if any(token in lowered for token in ("coin", "gem", "charm", "ring")):
        return ("$", "valuable", (245, 216, 118), None)

    return ("*", "valuable", None, None)


def _direction_target_xy(direction_action: str | None, origin: Position) -> tuple[int, int] | None:
    if direction_action in _CARDINAL_ACTION_DELTAS:
        dx, dy = _CARDINAL_ACTION_DELTAS[direction_action]
        return (origin.x + dx, origin.y + dy)
    if direction_action in _VECTOR_TO_ACTION.values():
        for (dx, dy), action in _VECTOR_TO_ACTION.items():
            if action == direction_action:
                return (origin.x + dx, origin.y + dy)
        return None


def _interaction_target_xy(direction_action: str | None, origin: Position) -> tuple[int, int]:
    target_xy = _direction_target_xy(direction_action, origin)
    if target_xy is not None:
        return target_xy
    return (origin.x, origin.y)


def _action_from_held_keys(
    held_directions: set[str],
    pressed_order: dict[str, int] | None = None,
) -> str | None:
    if pressed_order is None:
        pressed_order = {}

    use_recent_preference = bool(pressed_order)

    dx = 0
    dy = 0

    left_held = "move_left" in held_directions
    right_held = "move_right" in held_directions
    up_held = "move_up" in held_directions
    down_held = "move_down" in held_directions

    if left_held and right_held and use_recent_preference:
        left_order = pressed_order.get("move_left", -1)
        right_order = pressed_order.get("move_right", -1)
        dx = -1 if left_order >= right_order else 1
    elif left_held and not right_held:
        dx = -1
    elif right_held and not left_held:
        dx = 1

    if up_held and down_held and use_recent_preference:
        up_order = pressed_order.get("move_up", -1)
        down_order = pressed_order.get("move_down", -1)
        dy = -1 if up_order >= down_order else 1
    elif up_held and not down_held:
        dy = -1
    elif down_held and not up_held:
        dy = 1

    if dx == 0 and dy == 0:
        return None
    return _VECTOR_TO_ACTION.get((dx, dy))


def _find_interaction_npc(direction_action: str | None) -> int | None:
    player_ent = first_player_entity()
    if player_ent is None or not esper.has_component(player_ent, Position):
        return None

    player_pos = esper.component_for_entity(player_ent, Position)
    position_to_npc: dict[tuple[int, int], int] = {}
    for ent, (pos, _npc) in esper.get_components(Position, NPC):
        if esper.has_component(ent, Enemy):
            continue
        position_to_npc[(pos.x, pos.y)] = ent

    target_xy = _interaction_target_xy(direction_action, player_pos)
    return position_to_npc.get(target_xy)


def _find_interaction_creature(direction_action: str | None) -> int | None:
    """Any creature (friendly, wild, or hostile) on the faced tile. Used so the
    player can examine/interact with anything adjacent, not just friendlies."""
    player_ent = first_player_entity()
    if player_ent is None or not esper.has_component(player_ent, Position):
        return None

    player_pos = esper.component_for_entity(player_ent, Position)
    target_xy = _interaction_target_xy(direction_action, player_pos)
    for ent, (pos, _npc) in esper.get_components(Position, NPC):
        if (pos.x, pos.y) == target_xy:
            return ent
    return None


def _disposition_of(ent: int) -> str:
    if esper.has_component(ent, Player):
        return "You"
    if esper.has_component(ent, Enemy):
        return "Hostile"
    if esper.has_component(ent, Friendly):
        return "Friendly"
    if esper.has_component(ent, Deer):
        return "Wild animal"
    return "Neutral"


def _friendship_label(score: float) -> str:
    """A word for a friendship score, for status/dialogue screens."""
    if score >= 60:
        return "Close friend"
    if score >= 25:
        return "Friend"
    if score >= 5:
        return "Friendly"
    if score <= -60:
        return "Enemy"
    if score <= -25:
        return "Disliked"
    if score <= -5:
        return "Cold"
    return "Stranger"


def _player_talk(player_ent: int, npc_ent: int) -> str:
    """The player chats with a villager: nudge friendship both ways (via the same
    ``systems.interact`` the NPC social AI uses), pop a speech bubble above them,
    and return a one-line outcome for the dialogue footer."""
    clock = world_clock()
    turn = clock.turn if clock is not None else 0
    _delta_player, delta_npc = interact(player_ent, npc_ent, turn)
    score = 0.0
    if esper.has_component(player_ent, Relationships):
        score = friendship(esper.component_for_entity(player_ent, Relationships), npc_ent)
    indicator = "++" if delta_npc > 0 else ("--" if delta_npc < 0 else "~")
    return f"{indicator} friendship now {int(score)} ({_friendship_label(score)})"


def _creature_status_lines(game_map: GameMap, ent: int) -> list[str]:
    """Human-readable status block for any creature (the player, an NPC, a mob):
    name, disposition, needs, and active statuses."""
    lines = [
        f"Name: {entity_name(ent, fallback='Unknown')}",
        f"Disposition: {_disposition_of(ent)}",
    ]
    if esper.has_component(ent, Personality):
        traits = esper.component_for_entity(ent, Personality).traits
        lines.append("Traits: " + (", ".join(traits) if traits else "None"))
        player_ent = first_player_entity()
        if player_ent is not None and esper.has_component(ent, Relationships):
            rel = esper.component_for_entity(ent, Relationships)
            score = friendship(rel, player_ent)
            lines.append(f"Friendship: {int(score)} ({_friendship_label(score)})")
    if esper.has_component(ent, Needs):
        needs = esper.component_for_entity(ent, Needs)
        lines.append(f"Hunger: {int(needs.hunger)}%")
        lines.append(f"Thirst: {int(needs.thirst)}%")
        lines.append(f"Tiredness: {int(needs.tiredness)}%")

    statuses: list[str] = []
    if esper.has_component(ent, Position):
        pos = esper.component_for_entity(ent, Position)
        statuses = [status_label(name) for name in active_statuses(game_map, ent, pos)]
    lines.append("Status: " + (", ".join(statuses) if statuses else "Normal"))
    return lines


def _find_interaction_corpse(direction_action: str | None) -> int | None:
    player_ent = first_player_entity()
    if player_ent is None or not esper.has_component(player_ent, Position):
        return None

    player_pos = esper.component_for_entity(player_ent, Position)
    position_to_corpse: dict[tuple[int, int], int] = {
        (pos.x, pos.y): ent
        for ent, (pos, _corpse) in esper.get_components(Position, Corpse)
        if _entity_has_tradeable_items(ent)
    }

    target_xy = _interaction_target_xy(direction_action, player_pos)
    return position_to_corpse.get(target_xy)


def _npc_info_lines(game_map: GameMap, npc_ent: int) -> list[str]:
    dialogue_line = "..."
    if esper.has_component(npc_ent, Dialogue):
        dialogue_line = esper.component_for_entity(npc_ent, Dialogue).line

    inventory_count = 0
    if esper.has_component(npc_ent, Inventory):
        inventory_count = len(esper.component_for_entity(npc_ent, Inventory).items)

    equipped_count = 0
    if esper.has_component(npc_ent, Equipment):
        slots = esper.component_for_entity(npc_ent, Equipment).slots
        equipped_count = sum(1 for item in slots.values() if item)

    # Name / Disposition / Needs / Status, then dialogue-specific detail.
    return _creature_status_lines(game_map, npc_ent) + [
        f"Says: {dialogue_line}",
        f"Stock: {inventory_count} carried, {equipped_count} equipped",
    ]


@dataclass
class _TradeEntry:
    kind: str
    item_name: str
    slot_name: str | None = None
    item_index: int | None = None


def _list_trade_entries(actor_ent: int) -> list[_TradeEntry]:
    entries: list[_TradeEntry] = []

    if esper.has_component(actor_ent, Inventory):
        items = esper.component_for_entity(actor_ent, Inventory).items
        for idx, item_name in enumerate(items):
            entries.append(_TradeEntry(kind="inventory", item_name=item_name, item_index=idx))

    if esper.has_component(actor_ent, Equipment):
        slots = esper.component_for_entity(actor_ent, Equipment).slots
        for slot_name in sorted(slots.keys()):
            equipped_item = slots.get(slot_name)
            if equipped_item:
                entries.append(_TradeEntry(kind="equipment", item_name=equipped_item, slot_name=slot_name))

    return entries


def _remove_trade_entry(actor_ent: int, entry: _TradeEntry) -> str | None:
    if entry.kind == "inventory":
        if not esper.has_component(actor_ent, Inventory):
            return None
        inventory = esper.component_for_entity(actor_ent, Inventory)
        if entry.item_index is None:
            return None
        if entry.item_index < 0 or entry.item_index >= len(inventory.items):
            return None
        return inventory.items.pop(entry.item_index)

    if entry.kind == "equipment":
        if not esper.has_component(actor_ent, Equipment) or not entry.slot_name:
            return None
        equipment = esper.component_for_entity(actor_ent, Equipment)
        item = equipment.slots.get(entry.slot_name)
        if not item:
            return None
        equipment.slots[entry.slot_name] = None
        return item

    return None


def _ensure_inventory(actor_ent: int) -> Inventory:
    if esper.has_component(actor_ent, Inventory):
        return esper.component_for_entity(actor_ent, Inventory)

    inventory = Inventory(items=[])
    esper.add_component(actor_ent, inventory)
    return inventory


def _trade_item(source_ent: int, target_ent: int, entry: _TradeEntry) -> str:
    source_name = entity_name(source_ent)
    target_name = entity_name(target_ent)

    item_name = _remove_trade_entry(source_ent, entry)
    if item_name is None:
        return "Trade failed. Item was no longer available."

    target_inventory = _ensure_inventory(target_ent)
    target_inventory.items.append(item_name)
    return f"{source_name} traded {item_name} to {target_name}."


def _entity_has_tradeable_items(actor_ent: int) -> bool:
    if esper.has_component(actor_ent, Inventory):
        inventory = esper.component_for_entity(actor_ent, Inventory)
        if inventory.items:
            return True

    if esper.has_component(actor_ent, Equipment):
        slots = esper.component_for_entity(actor_ent, Equipment).slots
        if any(item for item in slots.values()):
            return True

    return False


def _loot_item_from_corpse(corpse_ent: int, player_ent: int, entry: _TradeEntry) -> str:
    corpse_name = entity_name(corpse_ent, fallback="Corpse")

    item_name = _remove_trade_entry(corpse_ent, entry)
    if item_name is None:
        return "Loot failed. Item was no longer available."

    player_inventory = _ensure_inventory(player_ent)
    player_inventory.items.append(item_name)

    if not _entity_has_tradeable_items(corpse_ent):
        return f"You looted {item_name} from {corpse_name}. Nothing else of value remains."

    return f"You looted {item_name} from {corpse_name}."


def _find_adjacent_feature(direction_action: str | None, component: type) -> int | None:
    """Return the entity carrying ``component`` on the tile the player is facing
    (the aimed direction), or ``None``. Used to interact with wells, stoves, and
    trees the same way corpses/NPCs are targeted."""
    player_ent = first_player_entity()
    if player_ent is None or not esper.has_component(player_ent, Position):
        return None

    player_pos = esper.component_for_entity(player_ent, Position)
    target_xy = _interaction_target_xy(direction_action, player_pos)
    for ent, (pos, _feature) in esper.get_components(Position, component):
        if (pos.x, pos.y) == target_xy:
            return ent
    return None


def _chop_tree(tree_ent: int, player_ent: int) -> str:
    """Chop one load of wood off a tree. When the last load is taken the tree
    falls (its entity is removed, freeing the tile)."""
    tree = esper.component_for_entity(tree_ent, Tree)
    inventory = _ensure_inventory(player_ent)
    inventory.items.append(WOOD)
    tree.wood -= 1
    if tree.wood <= 0:
        esper.delete_entity(tree_ent, immediate=True)
        return "You fell the tree, gathering a last piece of wood."
    return "You chop a piece of wood from the tree."


def _harvest_bush(bush_ent: int, player_ent: int) -> str:
    """Pick a ripe berry bush into the player's pack. The bush goes bare and its
    berries regrow a week later (handled by the growth processor)."""
    if pick_berries(bush_ent, world_clock()):
        _ensure_inventory(player_ent).items.append(BERRIES)
        return "You pick a handful of ripe berries."
    return "The bush has no ripe berries yet."


def _drink_from_well(well_ent: int, player_ent: int) -> str:
    """Drink from a well, quenching thirst. Wells are a renewable source."""
    if not esper.has_component(player_ent, Needs):
        return "The water is cool and clear."

    needs = esper.component_for_entity(player_ent, Needs)
    if needs.thirst <= 0:
        return "You drink from the well, but you were not thirsty."

    needs.thirst = 0.0
    return "You drink deeply from the well. Your thirst is quenched."


def _cook_at_stove(stove_ent: int, player_ent: int) -> str:
    """Turn one Wood + one Raw Meat into a Cooked Meat at the stove. Needs both:
    wood fuels the fire, raw meat is the ingredient."""
    if not esper.has_component(player_ent, Inventory):
        return "You have nothing to cook."

    inventory = esper.component_for_entity(player_ent, Inventory)
    raw_meat = next((item for item in inventory.items if is_raw_meat(item)), None)

    if raw_meat is None:
        return "You have no raw meat to cook. Butcher it from a corpse first."
    if WOOD not in inventory.items:
        return "The stove is cold. You need wood to make a fire."

    inventory.items.remove(WOOD)
    inventory.items.remove(raw_meat)
    inventory.items.append(cook_meat(raw_meat))
    return f"You light the stove and cook the {raw_meat.lower()} into a hot meal."


def _apply_consumable(player_ent: int, item_index: int) -> str | None:
    """Eat/drink the inventory item at ``item_index`` if it is consumable,
    removing it and reducing the matching need. Returns a message, or ``None``
    when the item is not food/drink (so the caller can fall back to equipping)."""
    if not esper.has_component(player_ent, Inventory):
        return None

    inventory = esper.component_for_entity(player_ent, Inventory)
    if item_index < 0 or item_index >= len(inventory.items):
        return None

    item_name = inventory.items[item_index]
    hunger_value = hunger_restored(item_name)
    thirst_value = thirst_restored(item_name)
    if hunger_value is None and thirst_value is None:
        return None

    if not esper.has_component(player_ent, Needs):
        esper.add_component(player_ent, Needs())
    needs = esper.component_for_entity(player_ent, Needs)

    inventory.items.pop(item_index)
    if hunger_value is not None:
        needs.hunger = max(0.0, needs.hunger - hunger_value)
        return f"You eat the {item_name}."
    needs.thirst = max(0.0, needs.thirst - thirst_value)
    return f"You drink the {item_name}."


def _bed_near_player() -> int | None:
    """The nearest Bed on or beside the player's tile (so the sleep action can bed
    them down rather than pitching a camp), or ``None`` if none is adjacent."""
    player_ent = first_player_entity()
    if player_ent is None or not esper.has_component(player_ent, Position):
        return None
    ppos = esper.component_for_entity(player_ent, Position)
    best: int | None = None
    best_dist = 2
    for ent, (pos, _bed) in esper.get_components(Position, Bed):
        dist = max(abs(pos.x - ppos.x), abs(pos.y - ppos.y))
        if dist <= 1 and dist < best_dist:
            best, best_dist = ent, dist
    return best


_CRAFT_MENU = [WOOD_WALL, WOOD_WINDOW, WOOD_DOOR]


def _craft_item(player_ent: int, item_name: str) -> str:
    """Craft one ``item_name`` from Wood in the player's pack, or explain why it
    can't be made. The crafted piece lands in the inventory to be placed later."""
    cost = craft_cost(item_name)
    if cost is None:
        return f"You don't know how to craft {item_name}."
    inventory = _ensure_inventory(player_ent)
    wood_count = inventory.items.count(WOOD)
    if wood_count < cost:
        return f"You need {cost} Wood to craft a {item_name} (you have {wood_count})."
    for _ in range(cost):
        inventory.items.remove(WOOD)
    inventory.items.append(item_name)
    return f"You craft a {item_name} from {cost} Wood."


def _place_buildable_at(
    player_ent: int, game_map: GameMap, item_name: str, target_xy: tuple[int, int]
) -> str:
    """Stake out a buildable piece onto ``target_xy`` as a **blueprint ghost**,
    consuming the item as its materials (the ghost is placed already-stocked, a
    bright blue "ready to raise" preview). Anybody -- you or a passing builder --
    then raises it into the real tile. Returns a log message."""
    tile = placed_tile(item_name)
    if tile is None:
        return f"You can't place the {item_name}."
    tx, ty = target_xy
    if not game_map.in_bounds(tx, ty):
        return "You can't build there."
    if tx == 0 or ty == 0 or tx == game_map.width - 1 or ty == game_map.height - 1:
        return "You can't build on the edge of the world."
    if game_map.tile_at(tx, ty) != game_map.FLOOR:
        return "You can only build on open ground."
    for _ent, (pos,) in esper.get_components(Position):
        if (pos.x, pos.y) == (tx, ty):
            return "Something is in the way."

    inventory = _ensure_inventory(player_ent)
    if item_name not in inventory.items:
        return f"You have no {item_name} to place."
    inventory.items.remove(item_name)
    esper.create_entity(
        Position(tx, ty),
        Renderable(tile, fg=_BLUEPRINT_READY_BLUE),
        Name("Blueprint"),
        Blueprint(tile=tile, stocked=True, site=None),
    )
    return f"You lay out a {item_name} blueprint. Face it and interact to raise it."


_BLUEPRINT_READY_BLUE = (96, 158, 240)


def _work_blueprint(ghost_ent: int, player_ent: int, game_map: GameMap) -> tuple[str, bool]:
    """The player works the blueprint they're facing: deposit a Wood to stock an
    unstocked ghost, or raise a stocked one into its real tile (finishing a cabin
    if it was the last piece). Building is labour, so a successful haul or raise
    **spends a turn** -- returns ``(message, took_turn)`` and only the no-op
    failures (nothing to build, or no wood to haul) come back free."""
    if not esper.has_component(ghost_ent, Blueprint):
        return "There is nothing to build here.", False
    bp = esper.component_for_entity(ghost_ent, Blueprint)
    if not bp.stocked:
        inventory = _ensure_inventory(player_ent)
        if WOOD not in inventory.items:
            return "You need Wood to stock this blueprint.", False
        inventory.items.remove(WOOD)
        stock_blueprint(ghost_ent)
        return "You haul wood to the blueprint. It's ready to raise.", True
    raise_blueprint(game_map, ghost_ent)
    return "You raise the piece into place.", True


_LOOK_TALK_RANGE = 5


_LOOK_TRADE_RANGE = 1


_LOOK_MELEE_RANGE = 1


def _chebyshev_from_player(player_pos: Position, x: int, y: int) -> int:
    return max(abs(x - player_pos.x), abs(y - player_pos.y))


def _creature_at_xy(x: int, y: int) -> int | None:
    """The character (NPC or the player) standing on world cell (x, y)."""
    for ent, (pos, _npc) in esper.get_components(Position, NPC):
        if (pos.x, pos.y) == (x, y):
            return ent
    for ent, (pos, _player) in esper.get_components(Position, Player):
        if (pos.x, pos.y) == (x, y):
            return ent
    return None


def _renderable_at_xy(x: int, y: int, skip: int | None = None) -> int | None:
    """Any drawable entity on (x, y) other than ``skip`` (a tree, corpse, item,
    well, ...). Used to name whatever the look cursor rests on."""
    for ent, (pos, _rend) in esper.get_components(Position, Renderable):
        if ent == skip:
            continue
        if (pos.x, pos.y) == (x, y):
            return ent
    return None


def _terrain_name(game_map: GameMap, x: int, y: int) -> str:
    tile = game_map.tile_at(x, y)
    return {
        game_map.WALL: "a wall",
        game_map.WATER: "water",
        game_map.DOOR: "a door",
        game_map.WINDOW: "a window",
        game_map.FLOOR: "open ground",
    }.get(tile, "the ground")


def _look_available_actions(target_ent: int, dist: int) -> list[str]:
    """The interaction verbs the look cursor offers for ``target_ent`` at the
    given distance. Status is always offered; talking reaches a few tiles; trade
    and melee only work when standing right next to the target."""
    if esper.has_component(target_ent, Player):
        return ["Status"]

    options: list[str] = []
    if dist <= _LOOK_TALK_RANGE and esper.has_component(target_ent, Dialogue):
        options.append("Talk")
    if dist <= _LOOK_TRADE_RANGE and esper.has_component(target_ent, Friendly):
        options.append("Trade")
    if dist <= _LOOK_MELEE_RANGE and (
        esper.has_component(target_ent, Enemy) or esper.has_component(target_ent, Deer)
    ):
        options.append("Attack")
    options.append("Status")
    return options


def _perform_look_attack(target_ent: int) -> None:
    """Strike an adjacent huntable creature from look mode, reusing the movement
    processor's melee/death sfx so it feels like a normal attack, then let a turn
    pass so the world reacts."""
    name = entity_name(target_ent, fallback="the creature")
    movement = esper.get_processor(MovementProcessor)
    queue_message(f"You attack {name}.")
    if movement is not None and movement.on_melee_attack is not None:
        movement.on_melee_attack()
    slay_entity(target_ent)
    if movement is not None and movement.on_enemy_death is not None:
        movement.on_enemy_death()
    esper.process(WAIT_ACTION)


