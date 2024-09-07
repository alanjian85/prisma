camera.pos = {1.0, 2.0, 4.5}
camera.center = {0.0, 1.0, 0.0}

panorama = ImageHdr.new("textures/panorama.hdr")
scene:set_env_map(panorama)

lucy = Model.load("models/lucy.obj")
scene:add(lucy)
