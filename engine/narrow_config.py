"""One fixed unsplit N16/K128/W2/S8 output/down geometry; no runtime search."""

SHAPES = {'output': (2560, 4096), 'down': (2560, 9728)}


def narrow_config(rows, family):
    if type(rows) is not int or not 1 <= rows <= 32:
        raise ValueError('narrow MMA requires exact rows in 1..32')
    if family not in SHAPES:
        raise ValueError('unsupported narrow projection family')
    n, k = SHAPES[family]
    return {'constants': {'ROWS': rows, 'CHANNELS': n, 'INNER': k,
                         'TILE_ROWS': max(16, 1 << (rows-1).bit_length()),
                         'TILE_CHANNELS': 16, 'TILE_INNER': 128},
            'warps': 2, 'stages': 8, 'fusion': True, 'grid': (160,)}
