import luisa

# Perspective Camera
@luisa.func
def generate_ray(camera, p):
    """Perspective projection

    Args:
        camera: struct
        p: camera-space coords in [-1,1]^2
    """
    forward = normalize(camera.target - camera.origin)
    right = normalize(cross(forward, camera.up))
    up_perp = cross(right, forward)
    p = p * tan(0.5 * camera.fov)
    direction = normalize(p.x * right - p.y * up_perp + forward)
    return luisa.make_ray(camera.origin, direction, 0.0, 1e30)

@luisa.func
def tent_warp(u, radius):
    """Tent reconstruciton filter (for inter-pixel independent importance sampling)

    Args:
        u (float or float vectors): input sample in U[0,1)
        radius (float): radius of filter

    Returns:
        float or float vectors: warped sample following tent distribution
    """
    # for vectors, select syntax is component-wise.
    return radius * (sqrt(2*u)-1) if u<0.5 else radius * (1-sqrt(2-2*u))
