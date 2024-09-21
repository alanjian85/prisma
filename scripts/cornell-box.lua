camera.pos = {-0.0, 0.0, -4.0}
camera.center = {0.0, 0.0, 0.0}
camera.fov = math.rad(40.0)

panorama = ImageHdr.new("textures/panorama.hdr")
scene:set_env_map(panorama)

cornell_box = Model.load("models/dragon.obj")
scene:add(cornell_box)
