from __future__ import annotations

import esper
import pytest
from uuid import uuid4


@pytest.fixture(autouse=True)
def reset_esper_world() -> None:
    test_world = f"test_{uuid4().hex}"
    esper.switch_world(test_world)
    esper.clear_database()
    yield
    esper.clear_database()
    esper.switch_world("default")
    esper.delete_world(test_world)
