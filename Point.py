class Point(object):
    """Creates a point on a coordinate plane with values x and y."""
    X: int
    Y: int

    def __init__(self, x, y):
        """Defines x and y variables"""
        self.X = x
        self.Y = y

    def __str__(self):
        return "Point(%s,%s)" % (self.X, self.Y)

    def __hash__(self):
        return hash((self.X, self.Y))

    def __eq__(self, other):
        return (self.X, self.Y) == (other.X, other.Y)

    def __ne__(self, other):
        # Not strictly necessary, but to avoid having both x==y and x!=y
        # True at the same time
        return not (self == other)
