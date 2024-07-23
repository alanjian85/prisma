camera.pos = {0.0, 7.0, 7.0}
camera.center = {0.0, 1.0, 0.0}
camera.up = {0.0, 1.0, 0.0}
camera.fov = math.rad(20.0)
camera.focus_dist = 1.0
camera.lens_angle = math.rad(0.0)

material_ground = Lambertian.new({0.4, 0.4, 0.4})
scene:add(Sphere.new({0.0, -1000.0, 0.0}, 1000.0, material_ground))

material1 = Dielectric.new(1.5)
scene:add(Sphere.new({0.0, 1.0, 0.0}, 1.0, material1))
