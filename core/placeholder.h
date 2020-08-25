#ifndef PLACEHOLDER
#define PLACEHOLDER
#include <iostream>

// class for allocating data
// TODO: ensure it outlives the tensors it allocates
template <class T>
class Allocator {
    uint64_t bytes_allocated;
    uint64_t bytes_unallocated;
    uint64_t bytes_currently_allocated;

    public:
        Allocator(): bytes_allocated(0),
                     bytes_unallocated(0),
                     bytes_currently_allocated(0)
        {}

        T* allocate(uint64_t count) {
            T* data = (T*)calloc(count, sizeof(T));

            bytes_allocated = count * sizeof(T);
            bytes_currently_allocated = bytes_allocated;

            return data;
        }

        T* deallocate(T* data) {
            free(data);
            data = nullptr;

            bytes_unallocated += bytes_currently_allocated;
            bytes_currently_allocated = 0;

            return data;
        }

        T* reallocate(T* data, uint64_t new_count) {
            free(data);
            data = (T*)calloc(new_count, sizeof(T));

            bytes_unallocated += bytes_currently_allocated;
            bytes_currently_allocated = new_count * sizeof(T);
            bytes_allocated += bytes_currently_allocated;

            return data;
        }

        void printStats() {
            std::cout << "bytes_allocated: " << bytes_allocated << std::endl
                      << "bytes_unallocated: " << bytes_unallocated << std::endl
                      << "bytes_currently_allocated: " << bytes_currently_allocated << std::endl;
        }
};

// placeholder in memory
// for a tensor in the graph
//
// in the graph, this will be the object
// to hold the data for a tensor
template <class T>
class Placeholder {
    T* data;
    uint64_t size;

    void fillData() {
        for (int i = 0; i < size; i++)
            data[i] = i*(i-1);
    }

    public:
        Placeholder(): size(0) {}

        Placeholder(Allocator<T>* allocator, uint64_t count) {
            size = count;
            data = allocator->allocate(count);
            fillData();
        }

        ~Placeholder() {
            if (data != nullptr)
                free(data);
        }

        void initialize(Allocator<T>* allocator, uint64_t new_size) {
            data = allocator->allocate(new_size);
            size = new_size;
            fillData();
        }

        void print() {
            for (int i = 0; i < size; i++)
                std::cout << data[i] << " ";
            std::cout << std::endl;
        }

        // naive print function
        // obviously ill-suited for higher dimensions
        // and sizes
        void print(std::vector<int> shape) {
            int r = *(shape.end()-2);
            int c = *(shape.end()-1);

            for (int i = 0; i < r; i++) {
                std::cout << "[ ";
                for (int j = 0; j < c; j++)
                    std::cout << data[i*c+j] << " ";

                std::cout << "]" << std::endl;
            }
        }
};
#endif
