class Battlefield:
    def __init__(self, bf_id: int, name: str = "", point_value: int = 1):
        self.bf_id = bf_id
        self.name = name or f"Battlefield {bf_id + 1}"
        self.point_value = point_value
        self.controller = None          # Player object or None
        self.units: dict = {}           # player_name -> list[UnitInstance]

    def get_units(self, player_name: str) -> list:
        return self.units.get(player_name, [])

    def set_units(self, player_name: str, units: list):
        self.units[player_name] = units

    def add_unit(self, player_name: str, unit):
        self.units.setdefault(player_name, []).append(unit)

    def remove_dead_units(self):
        for name in self.units:
            self.units[name] = [u for u in self.units[name] if u.is_alive]

    def is_contested(self, player1_name: str, player2_name: str) -> bool:
        return (
            bool(self.get_units(player1_name)) and
            bool(self.get_units(player2_name))
        )

    def all_units(self) -> list:
        return [u for units in self.units.values() for u in units]

    def clear_units(self):
        self.units.clear()

    def __repr__(self):
        controller = self.controller.name if self.controller else "None"
        return f"Battlefield({self.name}, ctrl={controller})"
