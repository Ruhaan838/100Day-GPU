#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

const int WIDTH = 1024;
const int HEIGHT = 768;

const float SPHERE_RADIUS = 0.5f;
const float SPERE_CENTER_X = 0.0f;
const float SPHERE_CENTER_Y = 0.0f;
const float SPHERE_CENTER_Z = -1.5f;

struct Vec3 {
    float x, y, z;

    __device__ Vec3(): x(0), y(0), z(0) {}
    __device__ Vec3(float x, float y, float z): x(x), y(y), z(z) {}

    __device__ Vec3 operator+(const Vec3& v) const {
        return Vec3(x + v.x, y + v.y, z + v.z);
    }

    __device__ Vec3 operator-(const Vec3& v) const {
        return Vec3(x - v.x, y - v.y, z - v.z);
    }

    __device__ Vec3 operator*(float scalar) const {
        return Vec3(x * scalar, y * scalar, z * scalar);
    }

    __device__ float dot(const Vec3& v) const {
        return x * v.x + y * v.y + z * v.z;
    }

    __device__ Vec3 normalize() const {
        float length = sqrtf(x * x + y * y + z * z);
        return Vec3(x / length, y / length, z / length);
    }
};

__global__ void render(unsigned char *image){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= WIDTH || y >= HEIGHT) return;

    int idx = (y * WIDTH + x) * 3; // 3 channels for RGB

    float u = (2.0f * x / WIDTH - 1.0f);
    float v = (2.0f * y / HEIGHT - 1.0f);

    Vec3 ray_origin(0.0f, 0.0f, 0.0f);
    Vec3 ray_dir(u, v, -1.0f);
    ray_dir = ray_dir.normalize();

    Vec3 sphere_center(SPERE_CENTER_X, SPHERE_CENTER_Y, SPHERE_CENTER_Z);

    Vec3 oc = ray_origin - sphere_center;
    float a = ray_dir.dot(ray_dir);
    float b = 2.0f * oc.dot(ray_dir);
    float c = oc.dot(oc) - SPHERE_RADIUS * SPHERE_RADIUS;
    float discriminant = b * b - 4 * a * c;

    if (discriminant >= 0){
        float t = (-b - sqrtf(discriminant)) / (2.0f * a);
        Vec3 hit_point = ray_origin + ray_dir * t;
        Vec3 normal = (hit_point - sphere_center).normalize();

        Vec3 light_dir(0.0f, 1.0f, -1.0f);
        light_dir = light_dir.normalize();
        float light_intensity = fmaxf(0.0f, normal.dot(light_dir));

        image[idx] = (unsigned char)(255 * light_intensity);
        image[idx + 1] = (unsigned char)(255 * light_intensity);
        image[idx + 2] = (unsigned char)(255 * light_intensity);
    } else {
        image[idx] = 0;
        image[idx + 1] = 0;
        image[idx + 2] = 0; 
    }
}

void save_img(unsigned char *image){
    FILE *f = fopen("output.ppm", "wb");
    fprintf(f, "P6\n%d %d\n255\n", WIDTH, HEIGHT);
    fwrite(image, 1, WIDTH * HEIGHT * 3, f);
    fclose(f);
}

int main(){
    unsigned char *d_image, *image;
    size_t image_size = WIDTH * HEIGHT * 3 * sizeof(unsigned char);
    cudaMalloc((void**)&d_image, image_size);
    image = (unsigned char*)malloc(image_size);

    dim3 block_size(16, 16);
    dim3 grid_size((WIDTH + block_size.x - 1) / block_size.x, (HEIGHT + block_size.y - 1) / block_size.y);
    render<<<grid_size, block_size>>>(d_image);
    cudaMemcpy(image, d_image, image_size, cudaMemcpyDeviceToHost);

    save_img(image);

    cudaFree(d_image);
    free(image);
    return 0;
}