use std::{cell::RefCell, error::Error, fs, rc::Rc};

use clap::Parser;
use prisma::{
    config::Config,
    core::{BindGroupLayoutSet, BindGroupSet, PostProcessor, RenderContext, Renderer},
    scripting::Scripting,
    textures::Textures,
};

fn build_scene(
    context: Rc<RenderContext>,
    config: &Config,
) -> Result<(BindGroupLayoutSet, BindGroupSet), Box<dyn Error>> {
    let textures = Rc::new(RefCell::new(Textures::new(context.clone())));

    let script = fs::read_to_string(&config.script)?;
    let scripting = Scripting::new(textures.clone())?;
    let scene = scripting.load(&script)?;

    let (texture_bind_group_layout, texture_bind_group) = textures.borrow().build();
    let (scene_bind_group_layout, scene_bind_group) = scene.build(&context)?;

    let bind_group_layout_set = BindGroupLayoutSet {
        texture: texture_bind_group_layout,
        scene: scene_bind_group_layout,
    };
    let bind_group_set = BindGroupSet {
        texture: texture_bind_group,
        scene: scene_bind_group,
    };
    Ok((bind_group_layout_set, bind_group_set))
}

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();
    let config = Config::parse();

    let context = Rc::new(pollster::block_on(RenderContext::new())?);
    let (bind_group_layout_set, bind_group_set) = build_scene(context.clone(), &config)?;

    let renderer = Renderer::new(context.clone(), &config, bind_group_layout_set);
    renderer.render(bind_group_set);
    let post_processor = PostProcessor::new(context.clone(), &config);
    post_processor.post_process(renderer.render_target());

    let image = pollster::block_on(post_processor.retrieve_result())?.unwrap();
    image.save(config.output)?;
    Ok(())
}
