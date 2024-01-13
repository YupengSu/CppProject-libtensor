#pragma once
#define THREAD_PER_BLOCK 256

namespace ts {
    extern void addMM(void* c, void* a, void* b, int size);
}
