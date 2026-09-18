from __future__ import annotations

import pytest

from zhishuxing.core.navigation import NavigationAdapter
from zhishuxing.webapp.app import create_app
from zhishuxing.webapp.service import ZhiShuXingWebService


@pytest.fixture(scope="session")
def navigation():
    adapter = NavigationAdapter()
    adapter.load_navigation(str(__import__("zhishuxing").config.paths.navigation_config))
    return adapter


@pytest.fixture()
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


@pytest.fixture(scope="session")
def service():
    return ZhiShuXingWebService()
