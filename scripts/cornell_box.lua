camera.pos = {278.0, 278.0, -800.0}
camera.center = {278.0, 278.0, 0.0}
camera.fov = math.rad(40.0)

red = Lambertian.new(Color2.new{0.65, 0.05, 0.05})
white = Lambertian.new(Color2.new{0.73, 0.73, 0.73})
green = Lambertian.new(Color2.new{0.12, 0.45, 0.15})
light = Light.new(Color2.new{15.0, 15.0, 15.0})

scene:add(Quad.new({555.0, 0.0, 0.0}, {0.0, 555.0, 0.0}, {0.0, 0.0, 555.0}, green))
scene:add(Quad.new({0.0, 0.0, 0.0}, {0.0, 555.0, 0.0}, {0.0, 0.0, 555.0}, red))
scene:add(Quad.new({343.0, 554.0, 332.0}, {-130.0, 0.0, 0.0}, {0.0, 0.0, -105.0}, light))
scene:add(Quad.new({0.0, 0.0, 0.0}, {555.0, 0.0, 0.0}, {0.0, 0.0, 555.0}, white))
scene:add(Quad.new({555.0, 555.0, 555.0}, {-555.0, 0.0, 0.0}, {0.0, 0.0, -555.0}, white))
scene:add(Quad.new({0.0, 0.0, 555.0}, {555.0, 0.0, 0.0}, {0.0, 555.0, 0.0}, white))
