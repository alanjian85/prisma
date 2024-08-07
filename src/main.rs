use std::error::Error;

use clap::Parser;
use prisma::{Config, Renderer};

fn main() -> Result<(), Box<dyn Error>> {
    let config = Config::parse();

    let image = pollster::block_on(async {
        let renderer = Renderer::new(&config).await;
        renderer.render();
        renderer.retrieve().await
    });

    image.save(config.output)?;

    Ok(())
}
