"""Fixed ordinary-decode narrow MMA configurations; no runtime search."""

SHAPES = {'qkv': (6144, 2560), 'output': (2560, 4096),
          'gateup': (19456, 2560), 'down': (2560, 9728)}


def narrow_config(rows, family):
    if type(rows) is not int or not 1 <= rows <= 32:
        raise ValueError('narrow MMA requires exact rows in 1..32')
    if family not in SHAPES:
        raise ValueError('unsupported narrow projection family')
    n,k = SHAPES[family]
    if family == 'qkv':
        nc,kc,warps,stages = ((16,128,2,5) if rows == 1 else
                              (32,64,2,8) if rows <= 16 else (32,128,2,5))
    elif family == 'output':
        nc,kc,warps,stages = 32,256,4,5
    elif family == 'gateup':
        nc,kc,warps,stages = ((16,128,1,5) if rows == 1 else
                              (64,128,4,3) if rows <= 16 else (32,128,2,3))
    else:
        nc,kc,warps,stages = ((32,512,4,4) if rows == 1 else
                              (32,512,4,5) if rows <= 16 else (32,256,4,5))
    return {'constants': {'ROWS': rows, 'CHANNELS': n, 'INNER': k,
                         'TILE_ROWS': max(16,1 << (rows-1).bit_length()),
                         'TILE_CHANNELS': nc, 'TILE_INNER': kc},
            'warps': warps, 'stages': stages, 'fusion': True,
            'grid': ((n+nc-1)//nc,)}
