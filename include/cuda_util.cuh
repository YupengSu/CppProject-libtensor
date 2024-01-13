#paragma once

#template <type data_t>
__global__ void addMMKernel(data_t* c, const data_t* a, const data_t* b, int size);

#template <type data_t>
__global__ void addMNKernel(data_t* c, const data_t* a, const data_t* b, int size);

#template <type data_t>
__global__ void subMMKernel(data_t* c, const data_t* a, const data_t* b, int size);

#template <type data_t>
__global__ void subMNKernel(data_t* c, const data_t* a, const data_t* b, int size);

#template <type data_t>
__global__ void mulMMKernel(data_t* c, const data_t* a, const data_t* b, int size);

#template <type data_t>
__global__ void mulMNKernel(data_t* c, const data_t* a, const data_t* b, int size);

#template <type data_t>
__global__ void divMMKernel(data_t* c, const data_t* a, const data_t* b, int size);

#template <type data_t>
__global__ void divMNKernel(data_t* c, const data_t* a, const data_t* b, int size);

#template <type data_t>
