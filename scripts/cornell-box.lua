camera.pos = {278.0, 278.0, -800.0}
camera.center = {278.0, 278.0, 0.0}
camera.fov = math.rad(40.0)

panorama = ImageHdr.new("textures/panorama.hdr")
scene:set_env_map(panorama)

cornell_box = Model.load("models/cornell-box.obj")
scene:add(cornell_box)
