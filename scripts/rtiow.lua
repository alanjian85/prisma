function length(v)
    return math.sqrt(v[1] * v[1] + v[2] * v[2] + v[3] * v[3])
end

math.randomseed(os.time())

camera.pos = {13.0, 2.0, 3.0}
camera.center = {0.0, 0.0, 0.0}
camera.up = {0.0, 1.0, 0.0}
camera.fov = math.rad(20.0)
camera.focus_dist = 10.0
camera.lens_angle = math.rad(0.6)

panorama = ImageHdr.new("textures/panorama.hdr")
scene:set_env_map(panorama)

material_ground = Lambertian.new({0.5, 0.5, 0.5})
scene:add(Sphere.new({0.0, -1000.0, 0.0}, 1000.0, material_ground))

for a = -11, 11 do
    for b = -11, 11 do
        local choose_mat = math.random()
        local center = {}
        center[1] = a + 0.9 * math.random()
        center[2] = 0.2
        center[3] = b + 0.9 * math.random()

        if length {center[1] - 4.0, center[2] - 0.2, center[3]} > 0.9 then
            if choose_mat < 0.8 then
                local albedo = {}
                albedo[1] = math.random() * math.random()
                albedo[2] = math.random() * math.random()
                albedo[3] = math.random() * math.random()
                local material = Lambertian.new(albedo)
                scene:add(Sphere.new(center, 0.2, material))
            elseif choose_mat < 0.95 then
                local albedo = {}
                albedo[1] = 0.5 * math.random() + 0.5
                albedo[2] = 0.5 * math.random() + 0.5
                albedo[3] = 0.5 * math.random() + 0.5
                local fuzz = 0.5 * math.random()
                local material = Metal.new(albedo, fuzz)
                scene:add(Sphere.new(center, 0.2, material))
            else
                local material = Dielectric.new(1.5)
                scene:add(Sphere.new(center, 0.2, material))
            end
        end
    end
end

material1 = Dielectric.new(1.5)
scene:add(Sphere.new({0.0, 1.0, 0.0}, 1.0, material1))

material2 = Lambertian.new({0.4, 0.2, 0.1})
scene:add(Sphere.new({-4.0, 1.0, 0.0}, 1.0, material2))

material3 = Metal.new({0.7, 0.6, 0.5}, 0.0)
scene:add(Sphere.new({4.0, 1.0, 0.0}, 1.0, material3))
