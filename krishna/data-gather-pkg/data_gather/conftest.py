

import pytest
from .DirectorRequestBuilder import DirectorRequestBuilder
from .BuilderPointImageryRequest import BuilderPointImageryRequest
from .pointImageryRequest import PointImageryRequest

@pytest.fixture
def testPointRequest():
    tempOrder = BuilderPointImageryRequest()
    director = DirectorRequestBuilder()
    director.construct(tempOrder)
    newOrder = tempOrder.request
    return(newOrder)


class Tests:

    def test_fixture(self):
        tcase = testPointRequest()
        assert isinstance(tcase, PointImageryRequest)
