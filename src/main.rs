use std::{error::Error, rc::Rc};

use clap::Parser;
use console::Emoji;
use glam::Vec3;
use prisma::{
    config::Config,
    render::{BindGroupLayoutSet, BindGroupSet, PostProcessor, RenderContext, Renderer},
    scene::{CameraBuilder, Scene},
};

fn build_scene(
    context: Rc<RenderContext>,
    config: &Config,
) -> Result<(BindGroupLayoutSet, BindGroupSet), Box<dyn Error>> {
    let (document, buffers, images) = gltf::import(&config.scene)?;

    let mut scene = Scene::new(context.clone());
    scene.load(&document.scenes().next().unwrap(), &buffers, &images);

    let hdri = scene.textures.load_texture_hdr(&config.hdri)?;
    scene.set_hdri(hdri);

    let camera = CameraBuilder::new()
        .pos(Vec3::new(3.5, 2.3, 3.3))
        .center(Vec3::new(0.0, 0.0, 0.0))
        .fov(25.0_f32.to_radians())
        .build(config.size.width, config.size.height);
    scene.set_camera(camera);

    let (scene_bind_group_layout, scene_bind_group) = scene.build(&context.clone())?;
    let (primitive_bind_group_layout, primitive_bind_group) = scene.primitives.build(&context)?;
    let (material_bind_group_layout, material_bind_group) = scene.materials.build(&context)?;
    let (texture_bind_group_layout, texture_bind_group) = scene.textures.build();

    let bind_group_layout_set = BindGroupLayoutSet {
        scene: scene_bind_group_layout,
        primitive: primitive_bind_group_layout,
        material: material_bind_group_layout,
        texture: texture_bind_group_layout,
    };
    let bind_group_set = BindGroupSet {
        scene: scene_bind_group,
        primitive: primitive_bind_group,
        material: material_bind_group,
        texture: texture_bind_group,
    };
    Ok((bind_group_layout_set, bind_group_set))
}

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();
    let config = Config::parse();

    let context = Rc::new(pollster::block_on(RenderContext::new())?);
    println!(
        "{} {} Parsing and loading the scene...",
        console::style("[1/4]").bold().dim(),
        Emoji("📜 ", "")
    );
    let (bind_group_layout_set, bind_group_set) = build_scene(context.clone(), &config)?;

    let renderer = Renderer::new(context.clone(), &config, bind_group_layout_set);
    println!(
        "{} {} Taking samples of path-traced rays...",
        console::style("[2/4]").bold().dim(),
        Emoji("📷 ", "")
    );
    renderer.render(bind_group_set);

    println!(
        "{} {} Applying post-processing effects...",
        console::style("[3/4]").bold().dim(),
        Emoji("🌟 ", "")
    );
    let post_processor = PostProcessor::new(context.clone(), &config);
    post_processor.post_process(renderer.render_target());

    let image = pollster::block_on(post_processor.retrieve_result())?.unwrap();
    println!(
        "{} {} Exporting the image...",
        console::style("[4/4]").bold().dim(),
        Emoji("🎞️  ", "")
    );
    image.save(config.output)?;
    Ok(())
}
