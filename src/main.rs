use std::error;

use vulkano::{
    instance::{Instance, InstanceCreateInfo},
    VulkanLibrary,
};

fn main() -> Result<(), Box<dyn error::Error>> {
    let library = VulkanLibrary::new()?;
    let instance = Instance::new(library, InstanceCreateInfo::default())?;
    for physical_device in instance.enumerate_physical_devices()? {
        println!("{}", physical_device.properties().device_name);
    }
    Ok(())
}
