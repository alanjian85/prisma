use std::fmt::{self, Display};
use std::str::FromStr;

use clap::Parser;

#[derive(Parser)]
#[command(version)]
pub struct Config {
    /// The image size of the output
    #[arg(short, long, default_value_t = Size::new(400, 225))]
    pub size: Size,

    /// The path to the output
    #[arg(short, long, default_value_t = String::from("output.png"))]
    pub output: String,

    /// The maximum depth of each camera ray
    #[arg(long, default_value_t = 50)]
    pub depth: u32,

    /// The number of samples per pixel
    #[arg(long, default_value_t = 1000)]
    pub samples: u32,

    /// The path to the Lua scene description
    pub script: String,
}

#[derive(Clone)]
pub struct Size {
    pub width: u32,
    pub height: u32,
}

impl Size {
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }
}

impl FromStr for Size {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        let size: Vec<_> = s.split('x').collect();
        if size.len() != 2 {
            return Err(String::from("invalid number of dimensions found in string"));
        }
        let width = size[0].parse().map_err(|e| format!("{e}"))?;
        let height = size[1].parse().map_err(|e| format!("{e}"))?;
        Ok(Self::new(width, height))
    }
}

impl Display for Size {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}x{}", self.width, self.height)
    }
}
