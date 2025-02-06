#include <iostream>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cmath>
#include <vector>
#include <chrono>

#define PI 3.14159265358979323846
 
// CUDA kernel to generate signals at each antenna element with phase delay
__global__ void generate_signals(cufftComplex *signals, float carrier_frequency, float D_delay, int num_antennas, int *data_switch) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_antennas) {
        // Calculate the phase delay for the signal at each antenna
        float phase = idx * D_delay * 2 * PI;
        // Modulate the carrier signal with calculated phase and apply data switch (binary modulation)
        signals[idx].x = cosf(2 * PI * carrier_frequency + phase) * data_switch[idx];
        signals[idx].y = sinf(2 * PI * carrier_frequency + phase) * data_switch[idx];
    }
}

// CUDA kernel for applying time-domain beamforming delays to each antenna signal
__global__ void time_domain_beamforming(cufftComplex *signals, const float *delays, int num_antennas) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_antennas) {
        // Apply a phase shift corresponding to the delay for beamforming
        float delay_phase = 2 * PI * delays[idx];
        signals[idx].x *= cosf(delay_phase);
        signals[idx].y *= sinf(delay_phase);
    }
}

// CUDA kernel for applying frequency-domain beamforming weights to the FFT of signals
__global__ void frequency_domain_beamforming(cufftComplex *freq_domain_signals, const cufftComplex *beamforming_weights, int num_antennas) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_antennas) {
        // Complex multiplication of FFT signals and beamforming weights
        float temp_x = freq_domain_signals[idx].x * beamforming_weights[idx].x - freq_domain_signals[idx].y * beamforming_weights[idx].y;
        float temp_y = freq_domain_signals[idx].x * beamforming_weights[idx].y + freq_domain_signals[idx].y * beamforming_weights[idx].x;
        freq_domain_signals[idx].x = temp_x;
        freq_domain_signals[idx].y = temp_y;
    }
}
// Error check macros for CUDA and cuFFT
#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        std::cerr << "gpuError: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
        if (abort) exit(code);
    }
}

// Macros for checking errors in cuFFT API calls
#define cufftErrorCheck(ans) { cufftAssert((ans), __FILE__, __LINE__); }
inline void cufftAssert(cufftResult code, const char *file, int line, bool abort=true) {
    if (code != CUFFT_SUCCESS) {
        std::cerr << "cufftError: " << static_cast<int>(code) << " " << file << " " << line << std::endl;
        if (abort) exit(code);
    }
}
// Main program
int main(int argc, char *argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <num_antennas> <carrier_frequency> <sampling_frequency> <D_delay>" << std::endl;
        return -1;
    }

    int num_antennas = std::stoi(argv[1]);
    float carrier_frequency = std::stof(argv[2]); // In Hz
    float sampling_frequency = std::stof(argv[3]); // In Hz
    float D_delay = std::stof(argv[4]); // Delay value in seconds
    
    // Allocate memory for signals, delays, and data switch on the device
    cufftComplex *d_signals;
    float *d_delays;
    int *d_data_switch;
    gpuErrorCheck( cudaMalloc(&d_signals, num_antennas * sizeof(cufftComplex)) );
    gpuErrorCheck( cudaMalloc(&d_delays, num_antennas * sizeof(float)) );
    gpuErrorCheck( cudaMalloc(&d_data_switch, num_antennas * sizeof(int)) );

    // Simulate delays and data switch values
    std::vector<float> delays(num_antennas);
    std::vector<int> data_switch(num_antennas);
    for (int i = 0; i < num_antennas; ++i) {
        delays[i] = i * D_delay; // Linear delay increment
        data_switch[i] = (i % 2 == 0) ? 1 : 0; // Example data switch pattern
    }
    gpuErrorCheck( cudaMemcpy(d_delays, delays.data(), num_antennas * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrorCheck( cudaMemcpy(d_data_switch, data_switch.data(), num_antennas * sizeof(int), cudaMemcpyHostToDevice) );

    // Calculate grid and block sizes
    int blockSize = 256;
    int numBlocks = (num_antennas + blockSize - 1) / blockSize;

    // Generate signals with phase delays and data switch
    generate_signals<<<numBlocks, blockSize>>>(d_signals, carrier_frequency, D_delay, num_antennas, d_data_switch);
    gpuErrorCheck( cudaPeekAtLastError() );
    gpuErrorCheck( cudaDeviceSynchronize() );

    // Setup cuFFT plan for forward and inverse transforms
    cufftHandle plan;
    cufftErrorCheck( cufftPlan1d(&plan, num_antennas, CUFFT_C2C, 1) );

    // Perform forward FFT on the generated signals
    cufftErrorCheck( cufftExecC2C(plan, d_signals, d_signals, CUFFT_FORWARD) );

    //  // Create and initialize beamforming weights
    std::vector<cufftComplex> beamforming_weights(num_antennas);
    for (int i = 0; i < num_antennas; ++i) {
        // Initialize beamforming weights as needed
        beamforming_weights[i].x = 1.0f; // Example: set to 1 for now
        beamforming_weights[i].y = 0.0f;
    }

    // Allocate and copy beamforming weights to device
    cufftComplex *d_beamforming_weights;
    gpuErrorCheck(cudaMalloc(&d_beamforming_weights, num_antennas * sizeof(cufftComplex)));
    gpuErrorCheck(cudaMemcpy(d_beamforming_weights, beamforming_weights.data(), num_antennas * sizeof(cufftComplex), cudaMemcpyHostToDevice));

    // Perform frequency-domain beamforming
    frequency_domain_beamforming<<<numBlocks, blockSize>>>(d_signals, d_beamforming_weights, num_antennas);
    gpuErrorCheck(cudaPeekAtLastError());
    gpuErrorCheck(cudaDeviceSynchronize());

    // Perform inverse FFT to get the beamformed signal back to time domain
    cufftErrorCheck(cufftExecC2C(plan, d_signals, d_signals, CUFFT_INVERSE));

  

    // Start timing the beamforming process
    auto start = std::chrono::high_resolution_clock::now();

    // Perform time-domain beamforming
    time_domain_beamforming<<<numBlocks, blockSize>>>(d_signals, d_delays, num_antennas);
    gpuErrorCheck( cudaPeekAtLastError() );
    gpuErrorCheck( cudaDeviceSynchronize() );

    // Perform inverse FFT to get the beamformed signal back to time domain
    cufftErrorCheck( cufftExecC2C(plan, d_signals, d_signals, CUFFT_INVERSE) );

    // Stop timing the beamforming process
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "Time-domain beamforming took " << elapsed.count() << " ms.\n";

    // Copy the beamformed signals back to host
    std::vector<cufftComplex> beamformed_signals(num_antennas);
    gpuErrorCheck( cudaMemcpy(beamformed_signals.data(), d_signals, num_antennas * sizeof(cufftComplex), cudaMemcpyDeviceToHost) );

    // Process output signals as required
    // For example, print the real part of the first 10 beamformed signals
    //for (int i = 0; i < std::min(10, num_antennas); ++i) {
       // std::cout << "Antenna " << i << ": Signal = " << beamformed_signals[i].x << " + " << beamformed_signals[i].y << "j" << std::endl;
    //}

   // for all the antennas  
   for (int i = 0; i < num_antennas; ++i) {
    std::cout << "Antenna " << i << ": Signal = " << beamformed_signals[i].x << " + " << beamformed_signals[i].y << "j" << std::endl;
   }

    // Cleanup
    cufftErrorCheck( cufftDestroy(plan) );
    gpuErrorCheck( cudaFree(d_signals) );
    gpuErrorCheck( cudaFree(d_delays) );
    gpuErrorCheck( cudaFree(d_data_switch) );

    return 0;
}


