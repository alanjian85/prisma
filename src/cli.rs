use clap::Parser;
use std::fmt::{self, Display};
use std::str::FromStr;

#[derive(Parser)]
#[command(version)]
pub struct Cli {
    /// The image size of the output
    #[arg(short, long, default_value_t = Size::new(1024, 768))]
    pub size: Size,

    /// The path to the output
    #[arg(short, long, default_value_t = String::from("output.png"))]
    pub output: String,
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

    fn from_str(s: &str) -> Result<Self, Self::Err> {
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
