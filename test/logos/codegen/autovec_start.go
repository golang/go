// asmcheck -goexperiment simd

package main

//go:noinline
func startAtOne(x []uint32, k uint32) {
	for i := 1; i < len(x); i++ {
		// amd64/v3:"VPADDD"
		// amd64/v4:"VPADDD"
		// arm64:"VADD V[0-9]+\\.S4"
		x[i] += k
	}
}

//go:noinline
func startAtVariable(x []uint32, start int, k uint32) int {
	if start < 0 {
		return start
	}
	i := start
	for ; i < len(x); i++ {
		// amd64/v3:"VPADDD"
		// amd64/v4:"VPADDD"
		// arm64:"VADD V[0-9]+\\.S4"
		x[i] += k
	}
	return i
}

//go:noinline
func startWithSource(dst, src []uint32, start, end int) int {
	if start < 0 || end > len(dst) || end > len(src) {
		return start
	}
	i := start
	for ; i < end; i++ {
		// amd64/v3:"VPADDD"
		// amd64/v4:"VPADDD"
		// arm64:"VADD V[0-9]+\\.S4"
		dst[i] += src[i]
	}
	return i
}

func main() {}
