from __future__ import annotations

import ecs
import pytest
from uuid import uuid4

from systems import reset_flora_queue
import wildlife


@pytest.fixture(autouse=True)
def reset_esper_world() -> None:
    test_world = f"test_{uuid4().hex}"
    ecs.switch_world(test_world)
    ecs.clear_database()
    # Module-level simulation state outlives the world otherwise: the flora's
    # regrow queue and every map's wildlife stocks would carry entity ids from a
    # previous test into one where those ids mean something else.
    reset_flora_queue()
    wildlife.detach()
    yield
    ecs.clear_database()
    reset_flora_queue()
    wildlife.detach()
    ecs.switch_world("default")
    ecs.delete_world(test_world)
