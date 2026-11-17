// asmcheck -goexperiment simd

package main

//go:noinline
func add8(x []int8, k int8) {
	for i := range x {
		// amd64/v1:-"VPADDB"
		// amd64/v2:-"VPADDB"
		// amd64/v3:"VPADDB"
		// amd64/v4:"VPADDB"
		// arm64:"VADD V[0-9]+\\.B16"
		x[i] += k
	}
}

//go:noinline
func add16(x []uint16, k uint16) {
	for i := range x {
		// amd64/v3:"VPADDW"
		// amd64/v4:"VPADDW"
		// arm64:"VADD V[0-9]+\\.H8"
		x[i] += k
	}
}

//go:noinline
func add32(x []int32, k int32) {
	for i := range x {
		// amd64/v3:"VPADDD"
		// amd64/v4:"VPADDD"
		// arm64:"VADD V[0-9]+\\.S4"
		x[i] += k
	}
}

//go:noinline
func add64(x []uint64, k uint64) {
	for i := range x {
		// amd64/v3:"VPADDQ"
		// amd64/v4:"VPADDQ"
		// arm64:"VADD V[0-9]+\\.D2"
		x[i] += k
	}
}

//go:noinline
func sub32(x []int32, k int32) {
	for i := range x {
		// amd64/v3:"VPSUBD"
		// amd64/v4:"VPSUBD"
		// arm64:"VSUB V[0-9]+\\.S4"
		x[i] -= k
	}
}

//go:noinline
func and32(x []uint32, k uint32) {
	for i := range x {
		// amd64/v3:"VPAND"
		// amd64/v4:"VPAND"
		// arm64:"VAND V[0-9]+\\.B16"
		x[i] &= k
	}
}

//go:noinline
func or32(x []uint32, k uint32) {
	for i := range x {
		// amd64/v3:"VPOR"
		// amd64/v4:"VPOR"
		// arm64:"VORR V[0-9]+\\.B16"
		x[i] |= k
	}
}

//go:noinline
func xor32(x []uint32, k uint32) {
	for i := range x {
		// amd64/v3:"VPXOR"
		// amd64/v4:"VPXOR"
		// arm64:"VEOR V[0-9]+\\.B16"
		x[i] ^= k
	}
}

//go:noinline
func mul16(x []int16, k int16) {
	for i := range x {
		// amd64/v3:"VPMULLW"
		// amd64/v4:"VPMULLW"
		// arm64:"VMUL V[0-9]+\\.H8"
		x[i] *= k
	}
}

//go:noinline
func mul32(x []uint32, k uint32) {
	for i := range x {
		// amd64/v3:"VPMULLD"
		// amd64/v4:"VPMULLD"
		// arm64:"VMUL V[0-9]+\\.S4"
		x[i] *= k
	}
}

//go:noinline
func addSlices(dst, src []int32) {
	if len(dst) > len(src) {
		return
	}
	for i := range dst {
		// amd64/v3:"VPADDD"
		// amd64/v4:"VPADDD"
		// arm64:"VADD V[0-9]+\\.S4"
		dst[i] += src[i]
	}
}

//go:noinline
func scalarCall(x int32) int32 { return x + 1 }

//go:noinline
func callLoop(x []int32) {
	for i := range x {
		// amd64/v3:-"VPADDD"
		// amd64/v4:-"VPADDD"
		// arm64:-"VADD V[0-9]+\\.S4"
		x[i] = scalarCall(x[i])
	}
}

//go:noinline
func dependentLoop(x []int32) {
	for i := 1; i < len(x); i++ {
		// amd64/v3:-"VPADDD"
		// amd64/v4:-"VPADDD"
		// arm64:-"VADD V[0-9]+\\.S4"
		x[i] += x[i-1]
	}
}

//go:noinline
func strideLoop(x []int32) {
	for i := 0; i < len(x); i += 2 {
		// amd64/v3:-"VPADDD"
		// amd64/v4:-"VPADDD"
		// arm64:-"VADD V[0-9]+\\.S4"
		x[i] += 7
	}
}

//go:noinline
func floatSum(x []float64) float64 {
	var sum float64
	for _, v := range x {
		// amd64/v3:-"VADDPD"
		// amd64/v4:-"VADDPD"
		// arm64:-"VFADD V[0-9]+\\.D2"
		sum += v
	}
	return sum
}

func main() {}
