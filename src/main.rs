use std::{cell::RefCell, error::Error, fs, rc::Rc};

use clap::Parser;
use console::Emoji;
use prisma::{
    config::Config,
    models::Models,
    render::{BindGroupLayoutSet, BindGroupSet, PostProcessor, RenderContext, Renderer},
    scripting::Scripting,
    textures::Textures,
};

fn build_scene(
    context: Rc<RenderContext>,
    config: &Config,
) -> Result<(BindGroupLayoutSet, BindGroupSet), Box<dyn Error>> {
    let textures = Rc::new(RefCell::new(Textures::new(context.clone())));
    let models = Rc::new(RefCell::new(Models::new()));

    let script = fs::read_to_string(&config.script)?;
    let scripting = Scripting::new(textures.clone(), models.clone())?;
    let mut scene = scripting.load(config, &script)?;

    let (texture_bind_group_layout, texture_bind_group) = textures.borrow().build();
    let (material_bind_group_layout, material_bind_group) =
        models.borrow().materials().build(&context)?;
    let (mesh_bind_group_layout, mesh_bind_group) = models.borrow().meshes().build(&context)?;
    let (scene_bind_group_layout, scene_bind_group) = scene.build(&context.clone())?;

    let bind_group_layout_set = BindGroupLayoutSet {
        texture: texture_bind_group_layout,
        material: material_bind_group_layout,
        mesh: mesh_bind_group_layout,
        scene: scene_bind_group_layout,
    };
    let bind_group_set = BindGroupSet {
        texture: texture_bind_group,
        material: material_bind_group,
        mesh: mesh_bind_group,
        scene: scene_bind_group,
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
