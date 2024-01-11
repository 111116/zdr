import luisa
from luisa.mathtypes import *
from math import pi
import numpy as np
import imageio


compensate_mis = False

AliasEntry = luisa.StructType(
    prob=float,
    alias=int
)

def create_alias_table(values):
    """Create alias table for sampling from a discrete distribution.
    Args:
        values: A list of non-negative values.
    Returns:
        A tuple of (table, pdf), where table is a list of AliasEntry,
        and pdf is a list of probability density function.
    """
    sum = 0.0
    for v in values:
        sum += abs(v)
    if sum == 0.0:
        n = len(values)
        pdf = [1.0 / n] * n
    else:
        pdf = [abs(v) / sum for v in values]
    ratio = len(values) / sum
    over = []
    under = []
    table = []
    for i, v in enumerate(values):
        p = float(v * ratio)
        table.append(AliasEntry(prob=p, alias=i))
        if p > 1.0:
            over.append(i)
        elif p < 1.0:
            under.append(i)
    while over and under:
        o = over.pop()
        u = under.pop()
        table[o].prob -= 1.0 - table[u].prob
        table[u].alias = o
        if table[o].prob > 1.0:
            over.append(o)
        elif table[o].prob < 1.0:
            under.append(o)
    for i in over:
        table[i] = AliasEntry(prob=1.0, alias=i)
    for i in under:
        table[i] = AliasEntry(prob=1.0, alias=i)
    return table, pdf


@luisa.func
def sample_alias_table(table, n, u_in, offset=0):
    """
    Args:
        table: luisa.buffer of AliasEntry.
        n: Length of the table.
        u_in: A random number in [0, 1).
        offset: Offset of the table.
    Returns:
        A tuple of (index, u), where index is the sampled index, and
        u is the remapped random number in [0, 1).
        TODO tuple is not supported yet
    """
    u = u_in * n
    i = clamp(int(u), 0, n - 1)
    u_remapped = fract(u)
    entry = table.read(i + offset)
    index = i if u_remapped < entry.prob else entry.alias
    uu = u_remapped / entry.prob if u_remapped < entry.prob else (u_remapped - entry.prob) / (1.0 - entry.prob)
    return index, uu


@luisa.func
def rgb_to_cie_y(rgb: float3):
    return 0.212671 * rgb.x + 0.715160 * rgb.y + 0.072169 * rgb.z


def load_envmap(heap, img):
    """Load environment map from image.
    Args:
        heap: A Luisa bindless array.
        img: A numpy array of shape (height, width, 3).
    """
    assert img.ndim == 3 and img.shape[2] == 4
    if img.shape[0] != img.shape[1]:
        if img.shape[1] == img.shape[0]*2:
            img = img.repeat(2, axis=0)
        else:
            # TODO due to a bug in lcpy, images are fucked up when its not square
            raise RuntimeError('envmap must be strictly 1:2 or 1:1')
    print(img.shape, img.dtype)
    tex = luisa.image2d(img)
    heap.emplace(23332, tex) # default filter & address mode
    heap.update()

    # prepare sample map
    sample_map_size = int2(1024, 512)
    pixel_count = sample_map_size.x * sample_map_size.y
    scale_map_buffer = luisa.Buffer(pixel_count, dtype=float)
    @luisa.func
    def generate_weight_map_kernel():
        pixel = dispatch_id().xy
        center = make_float2(pixel) + 0.5
        sum_weight = 0.0
        sum_scale = 0.0
        filter_radius = 1.0
        filter_step = 0.125
        n = int(ceil(filter_radius / filter_step))
        for dy in range(-n, n+1):
            for dx in range(-n, n+1):
                offset = make_float2(make_int2(dx, dy)) * filter_step
                uv = (center + offset) / make_float2(sample_map_size)
                # print(uv, 'sampled', heap.texture2d_sample(23332, uv))
                scale = rgb_to_cie_y(heap.texture2d_sample(23332, uv).xyz)
                sin_theta = sin(uv.y * pi)
                weight = exp(-4.0 * length_squared(offset))
                value = weight * min(scale * sin_theta, 1e8)
                sum_weight += weight
                sum_scale += value
        pixel_id = pixel.y * sample_map_size.x + pixel.x
        # print(pixel_id, sum_scale / sum_weight)
        scale_map_buffer.write(pixel_id, sum_scale / sum_weight)
    generate_weight_map_kernel(dispatch_size=(sample_map_size.x, sample_map_size.y))
    scale_map = scale_map_buffer.numpy()

    # scaleimg = scale_map.reshape((sample_map_size.y, sample_map_size.x))
    # imageio.imwrite('scale.exr', scaleimg)
    # quit()

    # compensate mis
    if compensate_mis:
        raise NotImplementedError('compensate_mis')
        # FIXME should be equal area projection
        average_scale = scale_map.mean()
        scale_map = (scale_map - average_scale).clip(a_min=0.0)
    # construct conditional alias table
    row_averages = []
    pdfs = []
    aliases = []
    print("loading envmap...")
    # TODO speed up create alias table
    # construct marginal alias table p(x|y)
    for i in range(sample_map_size.y):
        row = scale_map[i * sample_map_size.x : (i+1) * sample_map_size.x]
        row_averages.append(row.mean())
        alias_table, pdf_table = create_alias_table(row)
        pdfs.extend(pdf_table)
        aliases.extend(alias_table)
    # construct marginal alias table p(y)
    alias_table, pdf_table = create_alias_table(row_averages)
    aliases = alias_table + aliases
    for y in range(sample_map_size.y):
        for x in range(sample_map_size.x):
            pdfs[y * sample_map_size.x + x] *= pdf_table[y] * pixel_count
    # upload alias table
    # TODO speed up upload alias table
    alias_buffer = luisa.buffer(aliases)
    pdf_buffer = luisa.Buffer.from_array(np.array(pdfs, dtype=np.float32))
    heap.emplace(23330, alias_buffer)
    heap.emplace(23331, pdf_buffer)
    # luisa.synchronize()
    

@luisa.func
def sample_envmap(envmap, u):
    pass

@luisa.func
def uv_to_direction(uv: float2):
    phi = 2 * pi * (1 - uv.x)
    theta = pi * uv.y
    y = cos(theta)
    sin_theta = sin(theta)
    x = sin(phi) * sin_theta
    z = cos(phi) * sin_theta
    return theta, phi, normalize(float3(x, y, z))

@luisa.func
def direction_to_uv(dir: float3):
    theta = acos(dir.y)
    phi = atan2(dir.x, dir.z)
    return make_float2(1 - phi / (2 * pi), theta / pi)


# luisa.init()
# img = imageio.imread('assets/empty_workshop_4k.exr')
# # convert from 3 channel to 4 channel
# img = np.concatenate([img, np.ones_like(img[..., :1])], axis=-1)
# print(img.shape)
# tmpheap = luisa.BindlessArray()
# load_envmap(tmpheap, np.asarray(img[0:2048, 0:4096]).astype(np.float32))

