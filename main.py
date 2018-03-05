import math


_author_ = 'wombat'
_project_ = 'MySimplePythonApplication'


class Solver:
    def demo(self):
        a = 1
        b = 2
        c = 3
        d = b * 2 - 4 * a * c
        disc = math.sqrt(d)
        root1 = (-b + disc) / (2 * a)
        root2 = (-b - disc) / (2 * a)
        print(root1, root2)


Solver().demo()