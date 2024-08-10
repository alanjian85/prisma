use std::{error::Error, rc::Rc};

use clap::Parser;
use prisma::{
    config::Config,
    core::{BindGroupLayoutSet, BindGroupSet, RenderContext, Renderer},
    textures::{ImageHdr, Textures},
};

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();
    let config = Config::parse();

    let image = pollster::block_on(async {
        let context = RenderContext::new().await;

        let mut texture = Textures::new(&context);
        texture.set_env_map(Rc::new(
            ImageHdr::new(&context, "textures/panorama.hdr").unwrap(),
        ));
        let (texture_bind_group_layout, texture_bind_group) = texture.build();

        let bind_group_layout_set = BindGroupLayoutSet {
            texture: &texture_bind_group_layout,
        };

        let bind_group_set = BindGroupSet {
            texture: &texture_bind_group,
        };

        let renderer = Renderer::new(&context, &config, bind_group_layout_set);
        renderer.render(bind_group_set);
        renderer.retrieve_result().await
    });

    image.save(config.output)?;
    Ok(())
}
