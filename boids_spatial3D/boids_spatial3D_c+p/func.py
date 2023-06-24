import math

def vec3Length(x, y, z):
    return math.sqrt(x * x + y * y + z * z)


def vec3Mult(x, y, z, multiplier):
    x *= multiplier
    y *= multiplier
    z *= multiplier


def vec3Div(x, y, z, divisor):
    x /= divisor
    y /= divisor
    z /= divisor

def vec3Normalize(x, y, z):
    # Get the length
    length = vec3Length(x, y, z)
    vec3Div(x, y, z, length)

