--camera.pos = {0.0, 0.0, -5.0}
--camera.center = {0.0, 0.0, 0.0}
--camera.fov = math.rad(40.0)
--
--panorama = ImageHdr.new("textures/panorama.hdr")
--scene:set_env_map(panorama)
--
--lucy = Model.load("models/diamond.obj")
--scene:add(lucy)

camera.pos = {2.5, 1.0, -2.5}
camera.center = {0.0, 0.5, 0.0}
camera.fov = math.rad(60.0)

panorama = ImageHdr.new("textures/panorama.hdr")
scene:set_env_map(panorama)

lucy = Model.load("models/diamond.obj")
scene:add(lucy)
