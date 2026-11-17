// asmcheck -goexperiment simd

package main

//go:noinline
func faultGuardedMap(dst, src []uint32, delta uint32) {
	if len(dst) > len(src) {
		return
	}
	for i := range dst {
		// amd64/v1:-"CALL.*logosAutovecPanicOnFault" -"VPADDD"
		// amd64/v2:-"CALL.*logosAutovecPanicOnFault" -"VPADDD"
		// amd64/v3:"CALL.*logosAutovecPanicOnFault" "VPADDD"
		// amd64/v4:"CALL.*logosAutovecPanicOnFault" "VPADDD"
		// arm64:"CALL.*logosAutovecPanicOnFault" "VADD V[0-9]+\\.S4"
		dst[i] = src[i] + delta
	}
}

//go:noinline
//go:nosplit
func nosplitMap(dst []uint32, delta uint32) {
	for i := range dst {
		// amd64/v3:-"CALL.*logosAutovecPanicOnFault" -"VPADDD"
		// amd64/v4:-"CALL.*logosAutovecPanicOnFault" -"VPADDD"
		// arm64:-"CALL.*logosAutovecPanicOnFault" -"VADD V[0-9]+\\.S4"
		dst[i] += delta
	}
}

func main() {}
