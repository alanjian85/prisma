use std::{cell::RefCell, error::Error, fs, rc::Rc};

use clap::Parser;
use console::Emoji;
use prisma::{
    config::Config,
    core::{BindGroupLayoutSet, BindGroupSet, PostProcessor, RenderContext, Renderer},
    materials::Materials,
    scripting::Scripting,
    textures::Textures,
};

fn build_scene(
    context: Rc<RenderContext>,
    config: &Config,
) -> Result<(BindGroupLayoutSet, BindGroupSet), Box<dyn Error>> {
    let textures = Rc::new(RefCell::new(Textures::new(context.clone())));
    let materials = Rc::new(RefCell::new(Materials::new(context.clone())));

    let script = fs::read_to_string(&config.script)?;
    let scripting = Scripting::new(textures.clone(), materials.clone())?;
    let mut scene = scripting.load(config, &script)?;

    let (textures_bind_group_layout, textures_bind_group) = textures.borrow().build();
    let (materials_bind_group_layout, materials_bind_group) = materials.borrow().build()?;
    let (scene_bind_group_layout, scene_bind_group) = scene.build(&context.clone())?;

    let bind_group_layout_set = BindGroupLayoutSet {
        textures: textures_bind_group_layout,
        materials: materials_bind_group_layout,
        scene: scene_bind_group_layout,
    };
    let bind_group_set = BindGroupSet {
        textures: textures_bind_group,
        materials: materials_bind_group,
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
