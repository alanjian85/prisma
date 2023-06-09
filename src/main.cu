#include <cstdint>

#include <stb_image_write.h>

int main() {
    const int width = 256, height = 256;
    unsigned char framebuffer[width * height * 3];
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float r = static_cast<float>(x) / (width - 1);
            float g = static_cast<float>(y) / (height - 1);
            float b = 0.25f;
            framebuffer[(y * width + x) * 3 + 0] = r * 255;
            framebuffer[(y * width + x) * 3 + 1] = g * 255;
            framebuffer[(y * width + x) * 3 + 2] = b * 255;
        }
    }
    stbi_write_png("image.png", width, height, 3, framebuffer, 0);
    return 0;
}
