camera.fov = math.rad(40.0)
camera.pos = {278.0, 278.0, -800.0}
camera.center = {278.0, 278.0, 0.0}

panorama = ImageHdr.new("textures/panorama.hdr")
scene:set_env_map(panorama)

red = Lambertian.new({0.65, 0.05, 0.05})
white = Lambertian.new({0.73, 0.73, 0.73})
green = Lambertian.new({0.12, 0.45, 0.15})
light = Light.new({15.0, 15.0, 15.0})

scene:add(Triangle.new({555.0, 0.0, 0.0}, {555.0, 555.0, 0.0}, {555.0, 0.0, 555.0}, green))
scene:add(Triangle.new({555.0, 555.0, 0.0}, {555.0, 0.0, 555.0}, {555.0, 555.0, 555.0}, green))

scene:add(Triangle.new({0.0, 0.0, 0.0}, {0.0, 555.0, 0.0}, {0.0, 0.0, 555.0}, red))
scene:add(Triangle.new({0.0, 555.0, 0.0}, {0.0, 0.0, 555.0}, {0.0, 555.0, 555.0}, red))

scene:add(Triangle.new({343.0, 554.0, 332.0}, {213.0, 554.0, 332.0}, {343.0, 554.0, 227.0}, light))
scene:add(Triangle.new({213.0, 554.0, 332.0}, {343.0, 554.0, 227.0}, {213.0, 554.0, 227.0}, light))

scene:add(Triangle.new({0.0, 0.0, 0.0}, {555.0, 0.0, 0.0}, {0.0, 0.0, 555.0}, white))
scene:add(Triangle.new({555.0, 0.0, 0.0}, {0.0, 0.0, 555.0}, {555.0, 0.0, 555.0}, white))

scene:add(Triangle.new({555.0, 555.0, 555.0}, {0.0, 555.0, 555.0}, {555.0, 555.0, 0.0}, white))
scene:add(Triangle.new({0.0, 555.0, 555.0}, {555.0, 555.0, 0.0}, {0.0, 555.0, 0.0}, white))

scene:add(Triangle.new({0.0, 0.0, 555.0}, {555.0, 0.0, 555.0}, {0.0, 555.0, 555.0}, white))
scene:add(Triangle.new({555.0, 0.0, 555.0}, {0.0, 555.0, 555.0}, {555.0, 555.0, 555.0}, white))
