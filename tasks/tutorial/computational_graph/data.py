from expr_graphs import EXPRESSIONS


def compute_expression(expr_name, x, y, p):
    if expr_name == 'add':
        return (x + y) % p
    if expr_name == 'sub':
        return (x - y) % p
    if expr_name == 'xy':
        return (x * y) % p
    if expr_name == 'x2_y2':
        return (x * x + y * y) % p
    if expr_name == 'x2_xy_y2':
        return (x * x + x * y + y * y) % p
    if expr_name == 'x2_xy_y2_x':
        return (x * x + x * y + y * y + x) % p
    if expr_name == 'x3_xy':
        return (x * x * x + x * y) % p
    if expr_name == 'x3_xy2_y':
        return (x * x * x + x * y * y + y) % p
    raise ValueError(f'Unknown expression: {expr_name}. Available: {EXPRESSIONS}')

