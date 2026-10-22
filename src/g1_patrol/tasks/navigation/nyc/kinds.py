"""Registry of the tile kinds: street (:mod:`.tiles`), indoor (:mod:`.indoor`), the lab (:mod:`.makerlab`) and
digital twins of real places (:mod:`.twin`). The street kinds keep their indices."""

from collections.abc import Callable

from .indoor import INDOOR_BUILDERS, INDOOR_KINDS
from .makerlab import LAB_BUILDERS
from .tiles import BUILDERS, KINDS, TileLayout

ALL_BUILDERS: dict[str, Callable[..., TileLayout]] = {**BUILDERS, **INDOOR_BUILDERS, **LAB_BUILDERS}
TRAINING_KINDS = KINDS + INDOOR_KINDS
"""The street and indoor kinds the B2-W's general policy trains on."""
LAB_KINDS = list(LAB_BUILDERS)
"""Tiles of the lab the B2-W is deployed in."""
ALL_KINDS = TRAINING_KINDS + LAB_KINDS + ["twin"]
"""Every tile kind; "twin" is built from a description of a real place (:mod:`.twin`)."""
