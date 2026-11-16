package cgoprobe

/*
#include <stdint.h>
typedef struct { uint64_t a; uint64_t b; } pair;
static uint64_t fold_pair(pair p) { return p.a * UINT64_C(3) + p.b; }
static void bump(uint64_t *p) { *p += UINT64_C(1); }
*/
import "C"

func fold(a, b uint64) uint64 {
	return uint64(C.fold_pair(C.pair{a: C.uint64_t(a), b: C.uint64_t(b)}))
}

func increment(v uint64) uint64 {
	x := C.uint64_t(v)
	C.bump(&x)
	return uint64(x)
}
