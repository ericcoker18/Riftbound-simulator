"""
Spell and Gear effect resolution.

Parses ability text into structured effects and resolves them against game state.
Only covers effects that are simulatable without a full stack/targeting UI.
Complex effects (counter spells, hidden, etc.) are no-ops in simulation.
"""

import re
import random


# ---------------------------------------------------------------------------
# Effect data classes
# ---------------------------------------------------------------------------

class Effect:
    """Base class for all resolved effects."""
    pass


class DealDamage(Effect):
    def __init__(self, amount: int, target: str = "enemy_unit"):
        self.amount = amount
        self.target = target   # "enemy_unit", "all_enemy_units", "any_unit"


class ReadyUnit(Effect):
    def __init__(self, target: str = "friendly_unit"):
        self.target = target   # "friendly_unit", "all_friendly_units"


class BounceUnit(Effect):
    def __init__(self, target: str = "enemy_unit"):
        self.target = target   # "enemy_unit", "friendly_unit"


class DestroyUnit(Effect):
    def __init__(self, target: str = "enemy_unit"):
        self.target = target


class DrawCards(Effect):
    def __init__(self, amount: int):
        self.amount = amount


class BuffUnit(Effect):
    def __init__(self, amount: int, target: str = "friendly_unit", duration: str = "permanent"):
        self.amount = amount
        self.target = target
        self.duration = duration   # "permanent" or "turn"


class MoveUnit(Effect):
    def __init__(self, target: str = "friendly_unit"):
        self.target = target


class GearEquip(Effect):
    """Gear equip: attaches to a unit and grants a Might buff."""
    def __init__(self, might_bonus: int = 1):
        self.might_bonus = might_bonus


# ---------------------------------------------------------------------------
# Effect parser
# ---------------------------------------------------------------------------

def parse_effects(ability: str) -> list:
    """
    Parse an ability string into a list of Effect objects.
    Only handles the most common simulatable patterns.
    """
    if not ability:
        return []

    text = ability.lower()
    effects = []

    # Deal N to a unit / Deal N to all units
    for m in re.finditer(r"deal (\d+) to (all enemy units|all units|a unit|all friendly units|an enemy unit|a friendly unit)", text):
        n = int(m.group(1))
        scope = m.group(2)
        if "all enemy" in scope:
            effects.append(DealDamage(n, "all_enemy_units"))
        elif "all friendly" in scope:
            effects.append(DealDamage(n, "all_friendly_units"))
        elif "all" in scope:
            effects.append(DealDamage(n, "all_units"))
        else:
            effects.append(DealDamage(n, "enemy_unit"))

    # Deal N to an enemy / Deal N to each enemy
    for m in re.finditer(r"deal (\d+) to each enemy unit", text):
        effects.append(DealDamage(int(m.group(1)), "all_enemy_units"))

    # Ready a unit / Ready all friendly units
    if "ready all friendly units" in text or "friendly units enter ready" in text:
        effects.append(ReadyUnit("all_friendly_units"))
    elif re.search(r"ready a (friendly )?unit", text):
        effects.append(ReadyUnit("friendly_unit"))

    # Draw N
    for m in re.finditer(r"draw (\d+)", text):
        effects.append(DrawCards(int(m.group(1))))

    # Give a unit +N might
    for m in re.finditer(r"give (?:a|an|each|all)? ?(?:friendly )?units? \+(\d+)\s*(?::rb_might:|might)", text):
        n = int(m.group(1))
        if "each" in text[max(0, m.start()-5):m.start()+20] or "all" in text[max(0, m.start()-5):m.start()+20]:
            effects.append(BuffUnit(n, "all_friendly_units", "turn"))
        else:
            effects.append(BuffUnit(n, "friendly_unit", "turn"))

    # Return a unit to hand/base
    if re.search(r"return (?:an? )?(?:friendly |enemy )?unit.{0,20}(?:hand|base|owner)", text):
        if "enemy" in text:
            effects.append(BounceUnit("enemy_unit"))
        elif "friendly" in text:
            effects.append(BounceUnit("friendly_unit"))
        else:
            effects.append(BounceUnit("enemy_unit"))

    # Kill a unit / Destroy a unit
    if re.search(r"(?:kill|destroy) (?:a|an|target) (?:enemy )?unit", text):
        effects.append(DestroyUnit("enemy_unit"))

    # Equip (Gear): parse Might bonus from text
    if "[equip]" in text:
        m = re.search(r"\+(\d+)\s*(?::rb_might:|might)", text)
        bonus = int(m.group(1)) if m else 1
        effects.append(GearEquip(bonus))

    return effects


# ---------------------------------------------------------------------------
# Effect resolver
# ---------------------------------------------------------------------------

class EffectResolver:
    """
    Resolves parsed effects against the current game state.
    Uses simple heuristic targeting (no player input needed).
    """

    def resolve_spell(self, effects: list, caster, opponent, battlefields: list,
                       bf=None, strategy=None):
        """
        Resolve all effects from a spell.
        strategy: ExpertStrategy for smart targeting, or None for basic.
        """
        self._strategy = strategy
        for effect in effects:
            self._resolve(effect, caster, opponent, battlefields, bf)
        self._strategy = None

    def resolve_gear(self, effects: list, owner, battlefields: list,
                     strategy=None):
        """Resolve gear effects (equip to a unit)."""
        self._strategy = strategy
        for effect in effects:
            if isinstance(effect, GearEquip):
                target = self._pick_friendly_unit(owner, battlefields)
                if target:
                    target.buffs.append(effect.might_bonus)
        self._strategy = None

    # --- Internal ---

    def _resolve(self, effect, caster, opponent, battlefields, bf):
        if isinstance(effect, DealDamage):
            self._resolve_damage(effect, caster, opponent, battlefields, bf)

        elif isinstance(effect, ReadyUnit):
            self._resolve_ready(effect, caster, battlefields)

        elif isinstance(effect, DrawCards):
            for _ in range(effect.amount):
                caster.draw_card()

        elif isinstance(effect, BuffUnit):
            self._resolve_buff(effect, caster, battlefields)

        elif isinstance(effect, BounceUnit):
            self._resolve_bounce(effect, caster, opponent, battlefields)

        elif isinstance(effect, DestroyUnit):
            target = self._pick_enemy_unit(opponent, battlefields, bf)
            if target:
                target.current_health = 0

    def _resolve_damage(self, effect, caster, opponent, battlefields, bf):
        if effect.target == "all_enemy_units":
            for unit in self._all_units(opponent, battlefields):
                unit.current_health -= effect.amount
        elif effect.target == "all_friendly_units":
            for unit in self._all_units(caster, battlefields):
                unit.current_health -= effect.amount
        elif effect.target == "all_units":
            for unit in self._all_units(caster, battlefields) + self._all_units(opponent, battlefields):
                unit.current_health -= effect.amount
        else:
            # Single enemy unit — pick weakest (easiest to finish off)
            target = self._pick_enemy_unit(opponent, battlefields, bf)
            if target:
                target.current_health -= effect.amount

    def _resolve_ready(self, effect, caster, battlefields):
        if effect.target == "all_friendly_units":
            for unit in self._all_units(caster, battlefields) + caster.base_units:
                unit.is_exhausted = False
        else:
            target = self._pick_exhausted_friendly(caster, battlefields)
            if target:
                target.is_exhausted = False

    def _resolve_buff(self, effect, caster, battlefields):
        if "all" in effect.target:
            for unit in self._all_units(caster, battlefields):
                unit.buffs.append(effect.amount)
        else:
            target = self._pick_friendly_unit(caster, battlefields)
            if target:
                target.buffs.append(effect.amount)

    def _resolve_bounce(self, effect, caster, opponent, battlefields):
        if effect.target == "enemy_unit":
            # Pick weakest enemy unit and return it to their hand as a card
            target = self._pick_enemy_unit(opponent, battlefields, None)
            if target:
                # Remove from battlefield
                for bf in battlefields:
                    units = bf.get_units(opponent.name)
                    if target in units:
                        units.remove(target)
                        bf.set_units(opponent.name, units)
                        break
                # Return card to hand
                opponent.hand.append(target.card)
        elif effect.target == "friendly_unit":
            target = self._pick_friendly_unit(caster, battlefields)
            if target:
                for bf in battlefields:
                    units = bf.get_units(caster.name)
                    if target in units:
                        units.remove(target)
                        bf.set_units(caster.name, units)
                        break
                caster.hand.append(target.card)

    # --- Targeting helpers ---

    def _all_units(self, player, battlefields):
        units = []
        for bf in battlefields:
            units.extend(bf.get_units(player.name))
        return units

    def _pick_enemy_unit(self, opponent, battlefields, bf_hint):
        """Pick enemy unit to target. Delegates to strategy if available."""
        if self._strategy:
            return self._strategy.pick_damage_target(opponent, battlefields, bf_hint)
        candidates = []
        if bf_hint:
            candidates = bf_hint.get_units(opponent.name)
        if not candidates:
            candidates = self._all_units(opponent, battlefields)
        alive = [u for u in candidates if u.is_alive]
        if not alive:
            return None
        return min(alive, key=lambda u: u.current_health)

    def _pick_friendly_unit(self, caster, battlefields):
        """Pick friendly unit to buff/equip. Delegates to strategy if available."""
        if self._strategy:
            return self._strategy.pick_buff_target(caster, battlefields)
        candidates = self._all_units(caster, battlefields)
        alive = [u for u in candidates if u.is_alive]
        if not alive:
            return None
        return max(alive, key=lambda u: u.effective_might)

    def _pick_exhausted_friendly(self, caster, battlefields):
        """Pick exhausted friendly to ready. Delegates to strategy if available."""
        if self._strategy:
            return self._strategy.pick_ready_target(caster, battlefields)
        candidates = self._all_units(caster, battlefields) + caster.base_units
        exhausted = [u for u in candidates if u.is_exhausted and u.is_alive]
        if not exhausted:
            return None
        return random.choice(exhausted)
