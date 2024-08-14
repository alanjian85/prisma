camera.pos = {0.0, 7.0, 7.0}
camera.center = {0.0, 1.0, 0.0}
camera.fov = math.rad(15.0)
camera.focus_dist = 8.8
camera.lens_angle = math.rad(10.0)

panorama = ImageHdr.new("textures/panorama.hdr")
scene:set_env_map(panorama)

material_ground = Lambertian.new({0.4, 0.4, 0.4})
material_ball = Dielectric.new(1.5)

scene:add(Sphere.new({0.0, -1000.0, 0.0}, 1000.0, material_ground))
scene:add(Sphere.new({0.0, 1.0, 0.0}, 1.0, material_ball))
