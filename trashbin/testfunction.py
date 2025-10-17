def multiply(a, b):
    return a * b




#### Multiple A and B ####
def test_multiply():
    assert multiply(2, 3) == 6
    assert multiply(-1, 5) == -5
    assert multiply(0, 100) == 0
    assert multiply(-2, -3) == 6
    assert multiply(2.5, 4) == 10.0
def test_multiply_negative():
    assert multiply(-2, 3) == -6
    assert multiply(2, -3) == -6
    assert multiply(-2, -3) == 6