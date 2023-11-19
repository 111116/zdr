import luisa

# Perspective Camera
@luisa.func
def generate_ray(camera, p):
    forward = normalize(camera.target - camera.origin)
    right = normalize(cross(forward, camera.up))
    up_perp = cross(right, forward)
    p = p * tan(0.5 * camera.fov)
    direction = normalize(p.x * right - p.y * up_perp + forward)
    return luisa.make_ray(camera.origin, direction, 0.0, 1e30)
