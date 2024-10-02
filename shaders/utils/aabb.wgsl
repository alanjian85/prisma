struct Aabb3 {
    min: vec3f,
    max: vec3f,
}

fn aabb_intersect(aabb: Aabb3, ray: Ray, interval: Interval) -> bool {
    var t_interval = interval;
    for (var i = 0; i < 3; i++) {
        let inv_dir = 1.0 / ray.dir[i];

        let t0 = (aabb.min[i] - ray.orig[i]) * inv_dir;
        let t1 = (aabb.max[i] - ray.orig[i]) * inv_dir;

        if t0 < t1 {
            t_interval.min = max(t_interval.min, t0);
            t_interval.max = min(t_interval.max, t1);
        } else {
            t_interval.min = max(t_interval.min, t1);
            t_interval.max = min(t_interval.max, t0);
        }

        if t_interval.min >= t_interval.max {
            return false;
        }
    }
    return true;
}