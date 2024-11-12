def get_col_num(coords, center_x0):
    x0, y0, x1, y1 = coords
    return int(x0 >= center_x0)