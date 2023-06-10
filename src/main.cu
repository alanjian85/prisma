#include <stb_image_write.h>

const int tileSize = 16;

__global__ void render(unsigned char *framebuffer, int width, int height) {
    int nTilesX = (width + tileSize - 1) / tileSize;
    int x = blockIdx.x % nTilesX * tileSize + threadIdx.x % tileSize;
    int y = blockIdx.x / nTilesX * tileSize + threadIdx.x / tileSize;
    int index = y * width + x;
    float r = static_cast<float>(x) / (width - 1);
    float g = static_cast<float>(y) / (height - 1);
    float b = 0.25f;
    framebuffer[index * 3 + 0] = r * 255;
    framebuffer[index * 3 + 1] = g * 255;
    framebuffer[index * 3 + 2] = b * 255;
}

int main() {
    const int width = 256, height = 256;
    unsigned char *framebuffer;
    cudaMallocManaged(&framebuffer, width * height * 3);
    int nTiles = ((width + tileSize - 1) / tileSize) *
                 ((height + tileSize - 1) / tileSize);
    render<<<nTiles, tileSize * tileSize>>>(framebuffer, width, height);
    cudaDeviceSynchronize();
    stbi_write_png("image.png", width, height, 3, framebuffer, 0);
    cudaFree(framebuffer);
    return 0;
}
