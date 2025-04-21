'''
test Request class
'''

from perflowai.core import Request


def test_Request():
    # Create a mock request
    req = Request(
        req_id=1,
        input_len=10,
        output_len=100,
        start_time=0,
    )

    # Check if the request is created correctly
    assert req.req_id == 1
    assert req.input_len == 10
    assert req.output_len == 100
    assert req.start_time == 0