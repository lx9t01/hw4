
/* 
Based off work by Nelson, et al.
Brigham Young University (2010)

Adapted by Kevin Yuh (2015)

Modified by Jordan Bonilla and Matthew Cedeno (2016)
*/


#include <stdio.h>
#include <cuda.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cufft.h>
#include <time.h>
#include "ta_utilities.hpp"

#define PI 3.14159265358979




/* Check errors on CUDA runtime functions */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

// declare texture reference 
texture<float, 2, cudaReadModeElementType> texreference;

/* Check errors on cuFFT functions */
void gpuFFTchk(int errval){
    if (errval != CUFFT_SUCCESS){
        printf("Failed FFT call, error code %d\n", errval);
    }
}


/* Check errors on CUDA kernel calls */
void checkCUDAKernelError()
{
    cudaError_t err = cudaGetLastError();
    if  (cudaSuccess != err){
        fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
    } else {
        fprintf(stderr, "No kernel error detected\n");
    }

}

__global__ 
void cudaMultiplyKernel(cufftComplex *raw_data, 
                        unsigned int nAngles, unsigned int sinogram_width) {
    unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    while (thread_index < nAngles * sinogram_width) {
        unsigned int p = thread_index % sinogram_width; 
        if (p < sinogram_width / 2) {
            raw_data[thread_index].x = raw_data[thread_index].x * ((2.0 * p) / sinogram_width);
            raw_data[thread_index].y = raw_data[thread_index].y * ((2.0 * p) / sinogram_width);
        } else {
            raw_data[thread_index].x = raw_data[thread_index].x * ((2.0 * (sinogram_width - p)) / sinogram_width);
            raw_data[thread_index].y = raw_data[thread_index].y * ((2.0 * (sinogram_width - p)) / sinogram_width);

        }
        
        // raw_data[thread_index].x = raw_data[thread_index].x;
        
        // raw_data[thread_index].y = raw_data[thread_index].y;
        thread_index += blockDim.x * gridDim.x;
    }
}

__global__
void cudaTakeFloatKernel(const cufftComplex *dev_out_filter, 
                        float *dev_sinogram_float, const unsigned int nAngles, const unsigned int sinogram_width) {
    unsigned int thread_index = blockDim.x * blockIdx.x + threadIdx.x;
    while (thread_index < nAngles * sinogram_width) {
        dev_sinogram_float[thread_index] = dev_out_filter[thread_index].x;
        thread_index += blockDim.x * gridDim.x;
    }
}


__global__
void cudaBackProjKernel(float *output_dev, 
                        const unsigned int nAngles, 
                        const unsigned int sinogram_width,
                        const unsigned int width, 
                        const unsigned int height) {

    unsigned int thread_index = blockDim.x * blockIdx.x + threadIdx.x;

    while (thread_index < width * height) {
        int y0 = height/2 - thread_index / width;
        int x0 = thread_index % width;

        for (int i = 0; i < nAngles; ++i) {
            float sita = (float)i * PI / nAngles;
            float d, xi, yi, q;
            if (sita == 0) {
                d = x0;
            } else if (sita == PI / 2) {
                d = y0;
            } else {
                float m = -cos(sita)/sin(sita);
                q = -1/m;
                xi = (y0 - m * x0)/(q - m);
                yi = q * xi;
                d = sqrtf(xi * xi + yi * yi);
            }
            if ((q > 0 && xi < 0)||(q < 0 && xi > 0)) {
                output_dev[thread_index] += tex2D(texreference, sinogram_width/2-d, i); // ( , xindex, yindex)
            } else {
                output_dev[thread_index] += tex2D(texreference, sinogram_width/2+d, i);
                // output_dev[index] += dev_sinogram_float[(int)(i * sinogram_width + d + sinogram_width / 2)];
            }
        }
        thread_index += blockDim.x * gridDim.x;
    }

}



void cudaCallMultiplyKernel (const unsigned int blocks, 
                            const unsigned int threadsPerBlock,
                            cufftComplex *raw_data,
                            const unsigned int nAngles, 
                            const unsigned int sinogram_width) {
    cudaMultiplyKernel<<<blocks, threadsPerBlock>>>(raw_data, nAngles, sinogram_width);
}


void cudaCallTakeFloatKernel(const unsigned int nBlocks, 
                            const unsigned int threadsPerBlock, 
                            const cufftComplex *dev_out_filter, 
                            float *dev_sinogram_float, 
                            const unsigned int nAngles, 
                            const unsigned int sinogram_width) {
    cudaTakeFloatKernel<<<nBlocks, threadsPerBlock>>>(dev_out_filter, dev_sinogram_float, nAngles, sinogram_width);
}

void cudaCallBackProjKernel(const unsigned int nBlocks, 
                            const unsigned int threadsPerBlock, 
                            float *output_dev, 
                            const unsigned int nAngles, 
                            const unsigned int sinogram_width,
                            const unsigned int width, 
                            const unsigned int height) {
    cudaBackProjKernel<<<nBlocks, threadsPerBlock>>>(output_dev, nAngles, sinogram_width, width, height);
}




int main(int argc, char** argv){
    // These functions allow you to select the least utilized GPU
    // on your system as well as enforce a time limit on program execution.
    // Please leave these enabled as a courtesy to your fellow classmates
    // if you are using a shared computer. You may ignore or remove these
    // functions if you are running on your local machine.
    TA_Utilities::select_least_utilized_GPU();
    int max_time_allowed_in_seconds = 30;
    TA_Utilities::enforce_time_limit(max_time_allowed_in_seconds);

    // Begin timer and check for the correct number of inputs
    time_t start = clock();
    if (argc != 7){
        fprintf(stderr, "Incorrect number of arguments.\n\n");
        fprintf(stderr, "\nArguments: \n \
        < Input sinogram text file's name > \n \
        < Width or height of original image, whichever is larger > \n \
        < Number of angles in sinogram >\n \
        < threads per block >\n \
        < number of blocks >\n \
        < output text file's name >\n");
        exit(EXIT_FAILURE);
    }






    /********** Parameters **********/

    int width = atoi(argv[2]);
    int height = width;
    int sinogram_width = (int)ceilf( height * sqrt(2) );

    int nAngles = atoi(argv[3]);


    int threadsPerBlock = atoi(argv[4]);
    int nBlocks = atoi(argv[5]);


    /********** Data storage *********/


    // GPU DATA STORAGE
    cufftComplex *dev_sinogram_cmplx;
    float *dev_sinogram_float; 
    float* output_dev;  // Image storage


    cufftComplex *sinogram_host;

    size_t size_result = width*height*sizeof(float);
    float *output_host = (float *)malloc(size_result);




    /*********** Set up IO, Read in data ************/

    sinogram_host = (cufftComplex *)malloc(  sinogram_width*nAngles*sizeof(cufftComplex) );

    FILE *dataFile = fopen(argv[1],"r");
    if (dataFile == NULL){
        fprintf(stderr, "Sinogram file missing\n");
        exit(EXIT_FAILURE);
    }

    FILE *outputFile = fopen(argv[6], "w");
    if (outputFile == NULL){
        fprintf(stderr, "Output file cannot be written\n");
        exit(EXIT_FAILURE);
    }

    int j, i;

    for(i = 0; i < nAngles * sinogram_width; i++){
        fscanf(dataFile,"%f",&sinogram_host[i].x);
        sinogram_host[i].y = 0;
    }

    fclose(dataFile);


    /*********** Assignment starts here *********/

    /* TODO ok: Allocate memory for all GPU storage above, copy input sinogram
    over to dev_sinogram_cmplx. */
    // in texture memory:

    gpuErrchk(cudaMalloc((void**)&dev_sinogram_cmplx, nAngles * sinogram_width * sizeof(cufftComplex)));
    gpuErrchk(cudaMemcpy(dev_sinogram_cmplx, sinogram_host, nAngles * sinogram_width * sizeof(cufftComplex), cudaMemcpyHostToDevice));



    /* TODO 1 ok: Implement the high-pass filter:
        - Use cuFFT for the forward FFT
        - Create your own kernel for the frequency scaling.
        - Use cuFFT for the inverse FFT
        - extract real components to floats
        - Free the original sinogram (dev_sinogram_cmplx)

        Note: If you want to deal with real-to-complex and complex-to-real
        transforms in cuFFT, you'll have to slightly change our code above.
    */

    // // create the high pass filter vector

    // cufftComplex *filter_v = (cufftComplex*)malloc(sizeof(cufftComplex) * sinogram_width);
    // for (int i = 0; i < sinogram_width; ++i) {
    //     filter_v[i].x = 1 - abs((float)(2 * i - sinogram_width) / sinogram_width);
    //     filter_v[i].y = 0;
    // } // on freq domain

    // DATA storage
    // cufftComplex *dev_filter_v;
    // gpuErrchk(cudaMalloc((void**)&dev_filter_v, sizeof(cufftComplex) * sinogram_width));
    // gpuErrchk(cudaMemcpy(dev_filter_v, filter_v, sinogram_width * sizeof(cufftComplex), cudaMemcpyHostToDevice));
    // cufftComplex *dev_out_filter;
    // gpuErrchk(cudaMalloc((void**)&dev_out_filter, sizeof(cufftComplex) * sinogram_width * nAngles));

    cufftHandle plan;
    gpuFFTchk(cufftPlan1d(&plan, sinogram_width, CUFFT_C2C, nAngles));
    gpuFFTchk(cufftExecC2C(plan, dev_sinogram_cmplx, dev_sinogram_cmplx, CUFFT_FORWARD));

    // call the kernel to perform the filter
    cudaCallMultiplyKernel(nBlocks, threadsPerBlock, dev_sinogram_cmplx, nAngles, sinogram_width);
    checkCUDAKernelError();
    printf("finish filter\n");
    // inverse fft
    gpuFFTchk(cufftExecC2C(plan, dev_sinogram_cmplx, dev_sinogram_cmplx, CUFFT_INVERSE));
    // destroy the cufft plan
    gpuFFTchk(cufftDestroy(plan));

    // take the float
    gpuErrchk(cudaMalloc((void**)&dev_sinogram_float, nAngles * sinogram_width * sizeof(float)));
    cudaCallTakeFloatKernel(nBlocks, threadsPerBlock, dev_sinogram_cmplx, dev_sinogram_float, nAngles, sinogram_width);
    checkCUDAKernelError();
    // free dev_sinogram_cmplx
    gpuErrchk(cudaFree(dev_sinogram_cmplx));
    printf("finish fft\n");
    // gpuErrchk(cudaFree(dev_out_filter));

    /* TODO 2: Implement backprojection.
        - Allocate memory for the output image.
        - Create your own kernel to accelerate backprojection.
        - Copy the reconstructed image back to output_host.
        - Free all remaining memory on the GPU.
    */

    // first I think I will have to copy dev_sinogram_float from device to host;
    // but actually it's not necessary
    // float *host_sinogram_float = (float*)malloc(sizeof(float)*nAngles*sinogram_width);
    // gpuErrchk(cudaMemcpy(host_sinogram_float, dev_sinogram_float, sizeof(float)*nAngles*sinogram_width, cudaMemcpyDeviceToHost));
    // gpuErrchk(cudaFree(dev_sinogram_float));

    cudaArray *cArray;
    cudaChannelFormatDesc channel;

    channel = cudaCreateChannelDesc<float>();

    // allocate device memory for cuda array
    gpuErrchk(cudaMallocArray(&cArray, &channel, nAngles, sinogram_width));
    int bytes = sizeof(float) * nAngles * sinogram_width;
    gpuErrchk(cudaMemcpyToArray(cArray, 0, 0, dev_sinogram_float, bytes, cudaMemcpyDeviceToDevice)); 
    // copy float sinogram from gloabl memory to texture memory =(
    gpuErrchk(cudaFree(dev_sinogram_float));

    // set texture filter mode
    texreference.filterMode = cudaFilterModeLinear;

    // set texture address mode
    texreference.addressMode[0] = cudaAddressModeWrap; // necessary???
    texreference.addressMode[1] = cudaAddressModeClamp;

    // bind texture reference with cuda array
    gpuErrchk(cudaBindTextureToArray(texreference, cArray));

    // Allocate memory for the output image.
    gpuErrchk(cudaMalloc((void**)&output_dev, size_result));
    gpuErrchk(cudaMemset(output_dev, 0, size_result));

    // call kernel
    cudaCallBackProjKernel(nBlocks, threadsPerBlock, output_dev, nAngles, sinogram_width, width, height);
    checkCUDAKernelError();

    // unbind texture
    cudaUnbindTexture(texreference);

    // copy result out
    gpuErrchk(cudaMemcpy(output_host, output_dev, size_result, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(output_dev));
    gpuErrchk(cudaFreeArray(cArray));



    // Allocate memory for the output image.
    // gpuErrchk(cudaMalloc((void**)&output_dev, size_result));
    // gpuErrchk(cudaMemset(output_dev, 0, size_result));
    // // call back projection kernel
    // cudaCallBackProjKernel(nBlocks, threadsPerBlock, dev_sinogram_float, output_dev, nAngles, sinogram_width, width, height);
    // printf("finish back proj\n");
    // checkCUDAKernelError();
    // gpuErrchk(cudaMemcpy(output_host, output_dev, size_result, cudaMemcpyDeviceToHost));
    // gpuErrchk(cudaFree(dev_sinogram_float));
    // gpuErrchk(cudaFree(output_dev));


    /* Export image data. */

    for(j = 0; j < width; j++){
        for(i = 0; i < height; i++){
            fprintf(outputFile, "%e ",output_host[j*width + i]);
        }
        fprintf(outputFile, "\n");
    }


    /* Cleanup: Free host memory, close files. */

    free(sinogram_host);
    free(output_host);

    fclose(outputFile);
    printf("CT reconstruction complete. Total run time: %f seconds\n", (float) (clock() - start) / 1000.0);
    return 0;
}



