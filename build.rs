use std::{
    error::Error,
    fs::{self, File},
    io::Write,
};

const SHADER_DIR: &str = "shaders-generated";
const INCLUDE_PREFIX: &str = "///#include";

fn main() -> Result<(), Box<dyn Error>> {
    println!("cargo:rerun-if-changed=shaders/");

    let shader_files = ["render.wgsl", "post_process.wgsl"];

    std::fs::create_dir_all(SHADER_DIR)?;
    for file in shader_files {
        generate_shader(file)?;
    }

    Ok(())
}

fn generate_shader(file_name: &str) -> Result<(), Box<dyn Error>> {
    let path = format!("shaders/{}", file_name);
    let out_path = SHADER_DIR.to_string() + &format!("/{}", file_name);

    let source = match fs::read_to_string(&path) {
        Ok(source) => source,
        Err(e) => panic!("Failed to read file {}: {}", path, e),
    };

    let mut result = String::new();
    preprocess_shader(&source, &mut result);

    let mut file = File::create(out_path)?;
    file.write_all(result.as_bytes())?;

    Ok(())
}

fn preprocess_shader(source: &str, result: &mut String) {
    for line in source.lines() {
        if let Some(stripped) = line.strip_prefix(INCLUDE_PREFIX) {
            let include_file = stripped.trim().replace('"', "");
            let include_source = get_include_source(&include_file);
            preprocess_shader(&include_source, result);
        } else {
            result.push_str(line);
            result.push('\n');
        }
    }
}

fn get_include_source(file_name: &str) -> String {
    let path = format!("shaders/{}", file_name);
    match fs::read_to_string(&path) {
        Ok(source) => source,
        Err(e) => panic!("Failed to read file {}: {}", path, e),
    }
}
