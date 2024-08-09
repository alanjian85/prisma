use std::error::Error;

use clap::Parser;
use prisma::{config::Config, core::Renderer};

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();
    let config = Config::parse();

    let image = pollster::block_on(async {
        let renderer = Renderer::new(&config).await;
        renderer.render();
        renderer.retrieve().await
    });

    image.save(config.output)?;
    Ok(())
}
