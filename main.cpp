#include "stb_image.h"
#include "stb_image_write.h"

#include <CL/cl.h>
#include <CL/opencl.hpp>

#include <cstdio>
#include <iostream>
#include <thread>
#include <unordered_set>
#include <vector>

const std::string kernel_source = R"(
// Linear Congruential Generator with GCC parameters
int lcg(int state) {
    return (1103515245 * state + 12345) & ~(1 << 31);
}

// Fisher-Yates algorithm for shuffling an array
void shuffle_array(int* array, int n, int state) {    
    for(int i = n - 1; i >= 1; i--) {
        state = lcg(state);
        int j = state % (i + 1);

        int tmp = array[i];
        array[i] = array[j];
        array[j] = tmp;
    }
}

__kernel void transform_pixel(__global unsigned char* input, int width, int height, __global unsigned char* share, int share_number, __global int* s0, __global int* s1, int prng_seed) {
    int id = get_global_id(0);
    prng_seed += 31 * id;

    const int subpixel_scale = 1 << ((k - 1) / 2);
    const int num_subpixels = 1 << (k - 1);

    int y = id / width;
    int x = id % width;
    int cols[num_subpixels];
    int rows[num_subpixels];

    for(int i = 0; i < num_subpixels; i++) {
        cols[i] = i;
        rows[i] = i;
    }

    shuffle_array(cols, num_subpixels, prng_seed);
    shuffle_array(rows, k, prng_seed);

    __global int* s = input[y * width + x] < 127 ? s0 : s1;

    for(int dy = 0; dy < subpixel_scale; dy++) {
        for(int dx = 0; dx < subpixel_scale; dx++) {
            share[(y * subpixel_scale + dy) * width * subpixel_scale + x * subpixel_scale + dx] = s[cols[(dy * subpixel_scale + dx)] + rows[share_number] * num_subpixels] * 255;
        }
    }
}
)";

// calculate the list of vectors J0 for the given k
std::vector<std::vector<int>> gen_j0(int k) {
    std::vector<std::vector<int>> j0(k, std::vector<int>(k, 0));

    for(int i = 0; i < k - 1; i++) {
        j0[i][i] = 1;
    }

    for(int i = 0; i < k - 1; i++) {
        j0[k - 1][i] = 1;
    }

    return j0;
}

// calculate dot product over GF(2) of j and a binary vector represented by x
int binary_dot(std::vector<int> const& j, int x) {
    int k = j.size();
    int result = 0;

    for(int i = 0; i < k; i++) {
        result += j[i] & (x >> (k - i - 1));
    }

    return result & 1;
}

// convert a binary vector to an integer
int column_to_int(std::vector<int> const& column) {
    int result = 0;

    for(int i = 0; i < column.size(); i++) {
        result <<= 1;
        result |= column[i];
    }
    
    return result;
}

// calculate the matrix S0 for the given k
std::vector<int> gen_s0(int k) {
    auto j0 = gen_j0(k);

    std::unordered_set<int> used_columns{};
    std::vector<int> s0(k * (1 << (k - 1)));
    std::vector<int> column(k);

    for(int x = 0; x < (1 << k); x++) {
        for(int i = 0; i < k; i++) {
            column[i] = binary_dot(j0[i], x);
        }

        int as_int = column_to_int(column);

        if(used_columns.find(as_int) == used_columns.end()) {
            used_columns.insert(as_int);
            
            for(int i = 0; i < k; i++) {
                s0[(used_columns.size() - 1) + i * (1 << (k - 1))] = column[i];
            }
        }
    }

    return s0;
}

// calculate the matrix S1 given the matrix S0
std::vector<int> s1_from_s0(std::vector<int> const& s0) {
    std::vector<int> s1(s0.size());

    for(int i = 0; i < s0.size(); i++) {
        s1[i] = !s0[i];
    }

    return s1;
}

int main(int argc, char** argv) {
    if(argc != 3 && argc != 4) {
        std::cout << "Usage: " << argv[0] << " <file> <number of shares> [OpenCL platform number (default 1)]" << std::endl;
        return 1;
    }

    char* filename = argv[1];
    int num_shares;

    if(sscanf(argv[2], "%d ", &num_shares) != 1) {
        std::cout << "Not a valid number: `" << argv[2] << "`" << std::endl;
        return 1;
    }

    int w;
    int h;
    unsigned char* img = stbi_load(filename, &w, &h, NULL, 1);
    int subpixel_scale = 1 << ((num_shares - 1) / 2);

    if(img == NULL) {
        std::cout << "Error loading image" << std::endl;
        return 1;
    }

    std::cout << "Image loaded successfully" << std::endl;

    auto s0 = gen_s0(num_shares);
    auto s1 = s1_from_s0(s0);

    std::vector<cl::Platform> allPlatforms{};
    cl::Platform::get(&allPlatforms);

    if(allPlatforms.size() == 0) {
        std::cout << "No platform found" << std::endl;
        return 1;
    }

    cl::Platform platform;

    if(allPlatforms.size() == 1) {
        platform = allPlatforms[0];
    } else if(argc == 3) {
        std::cout << "Found platforms: " << std::endl;

        for(int i = 0; i < allPlatforms.size(); i++) {
            std::cout << (i + 1) << ". " << allPlatforms[i].getInfo<CL_PLATFORM_NAME>() << std::endl;
        }

        int platformNum = 0;

        while(platformNum < 1 || platformNum > allPlatforms.size()) {
            std::cout << "Choose your platform (" << 1 << "-" << allPlatforms.size() << "): ";
            std::cin >> platformNum;
        }

        platform = allPlatforms[platformNum - 1];
    } else {
        int platformNum;

        if(sscanf(argv[3], "%d ", &platformNum) != 1) {
            std::cout << "Not a valid number: `" << argv[3] << "`" << std::endl;
            return 1;
        }

        if(platformNum < 1 || platformNum > allPlatforms.size()) {
            std::cout << "Platform number should be between 1 and " << allPlatforms.size() << std::endl;
            return 1;
        }

        platform = allPlatforms[platformNum - 1];
    }

    std::cout << "Using platform: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
    
    std::vector<cl::Device> all_devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if(all_devices.size() == 0) {
        std::cout << "No device found" << std::endl;
        return 1;
    }

    cl::Device device = all_devices[0];
    std::cout << "Using device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

    cl::Context context(device);
    cl::CommandQueue queue(context, device);

    cl::Buffer input(context, CL_MEM_READ_ONLY, w * h);

    std::vector<cl::Buffer> shares_bufs(num_shares);
    for(int i = 0; i < num_shares; i++) {
        shares_bufs[i] = cl::Buffer(context, CL_MEM_WRITE_ONLY, w * h * subpixel_scale * subpixel_scale);
    }

    cl::Buffer s0_buf(context, CL_MEM_READ_ONLY, s0.size() * sizeof(int));
    cl::Buffer s1_buf(context, CL_MEM_READ_ONLY, s1.size() * sizeof(int));

    queue.enqueueWriteBuffer(input, CL_TRUE, 0, w * h, img);
    queue.enqueueWriteBuffer(s0_buf, CL_TRUE, 0, s0.size() * sizeof(int), s0.data());
    queue.enqueueWriteBuffer(s1_buf, CL_TRUE, 0, s1.size() * sizeof(int), s1.data());

    stbi_image_free(img);

    cl::Program::Sources sources{};
    sources.push_back(kernel_source);
    cl::Program program(context, sources);
    char compileArgs[64];
    sprintf(compileArgs, "-Dk=%d", num_shares);

    if(program.build(device, compileArgs) != CL_SUCCESS) {
        std::cout << "Couldn't build program: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
        return 1;
    }

    cl::compatibility::make_kernel<cl::Buffer, int, int, cl::Buffer, int, cl::Buffer, cl::Buffer, int> generate_shares(cl::Kernel(program, "transform_pixel"));
    cl::NDRange global(w * h);
    std::vector<cl::Event> events(num_shares);

    int prng_seed = time(0);
    for(int i = 0; i < num_shares; i++) {
        events[i] = generate_shares(cl::EnqueueArgs(queue, global), input, w, h, shares_bufs[i], i, s0_buf, s1_buf, prng_seed);
    }

    std::cout << "GPU calculations queued" << std::endl;
    for(int i = 0; i < num_shares; i++) {
        events[i].wait();
    }
    std::cout << "GPU calculations done" << std::endl;

    std::vector<unsigned char*> output(num_shares);

    for(int i = 0; i < output.size(); i++) {
        output[i] = new unsigned char[w * h * subpixel_scale * subpixel_scale];
    }

    std::vector<std::thread> writing_threads{};

    for(int i = 0; i < num_shares; i++) {
        queue.enqueueReadBuffer(shares_bufs[i], CL_TRUE, 0, w * h * subpixel_scale * subpixel_scale, output[i]);
        writing_threads.push_back(std::thread([=] () {
            char filename[64];
            sprintf(filename, "share_%d.jpg", i);
            stbi_write_jpg(filename, w * subpixel_scale, h * subpixel_scale, 1, output[i], 70);
        }));
    }

    for(int i = 0; i < num_shares; i++) {
        writing_threads[i].join();
    }

    for(int i = 0; i < output.size(); i++) {
        delete[] output[i];
    }

    return 0;
}