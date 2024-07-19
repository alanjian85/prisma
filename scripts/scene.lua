camera.pos = {-2.0, 2.0, 1.0}
camera.center = {0.0, 0.0, -1.0}
camera.up = {0.0, 1.0, 0.0}
camera.fov = math.rad(20.0)
camera.focus_dist = 3.4
camera.lens_angle = math.rad(10.0)

material_ground = Lambertian.new({0.8, 0.8, 0.0})
material_center = Lambertian.new({0.1, 0.2, 0.5})
material_left = Dielectric.new(1.5)
material_bubble = Dielectric.new(1.0 / 1.5)
material_right = Metal.new({0.8, 0.6, 0.2}, 1.0)

scene:add(Sphere.new({0.0, -100.5, -1.0}, 100.0, material_ground))
scene:add(Sphere.new({0.0, 0.0, -1.2}, 0.5, material_center))
scene:add(Sphere.new({-1.0, 0.0, -1.0}, 0.5, material_left))
scene:add(Sphere.new({-1.0, 0.0, -1.0}, 0.4, material_bubble))
scene:add(Sphere.new({1.0, 0.0, -1.0}, 0.5, material_right))
