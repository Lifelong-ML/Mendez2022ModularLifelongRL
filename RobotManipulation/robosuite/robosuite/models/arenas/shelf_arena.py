import numpy as np
from robosuite.models.arenas import Arena
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.utils.mjcf_utils import array_to_string


class ShelfArena(Arena):
    """
    Workspace that contains two bins placed side by side.

    Args:
        shelf_pos (3-tuple): (x,y,z) position to place the shelf
        table_full_size (3-tuple): (L,W,H) full dimensions of the table
        table_friction (3-tuple): (sliding, torsional, rolling) friction parameters of the table
    """

    def __init__(
        self, shelf_pos=(0.1, -0.5, 0.8), table_full_size=(0.39, 0.49, 0.82), table_friction=(1, 0.005, 0.0001)
    ):
        super().__init__(xml_path_completion("arenas/shelf_arena.xml"))

        self.table_full_size = np.array(table_full_size)
        self.table_half_size = self.table_full_size / 2
        self.table_friction = table_friction

        self.shelf_body = self.worldbody.find("./body[@name='shelf']")
        self.table_top_abs = np.array(shelf_pos)

        self.configure_location()

    def configure_location(self):
        """Configures correct locations for this arena"""
        self.floor.set("pos", array_to_string(self.bottom_pos))
