camera.pos = {0.0, 0.0, 6.0}
camera.center = {0.0, 0.0, 0.0}
camera.fov = math.rad(20.0)

scene:set_env(Color3.new{1.0, 1.0, 1.0})

material_ball = Lambertian.new(Image.new("assets/earthmap.jpg"))
scene:add(Sphere.new({0.0, 0.0, 0.0}, 1.0, material_ball))
