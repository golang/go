// asmcheck -goexperiment simd

package main

type word uint32

//go:noinline
func namedLoop(x []word, k word) {
	for i := range x {
		// amd64/v3:"VPADDD"
		// amd64/v4:"VPADDD"
		// arm64:"VADD V[0-9]+\\.S4"
		x[i] += k
	}
}

//go:noinline
func finalIndex(x []uint32, n int, k uint32) int {
	if n > len(x) {
		return -1
	}
	i := 0
	for ; i < n; i++ {
		// amd64/v3:"VPADDD"
		// amd64/v4:"VPADDD"
		// arm64:"VADD V[0-9]+\\.S4"
		x[i] += k
	}
	return i
}

//go:noinline
func threeSlices(dst, a, b []uint32) {
	if len(dst) > len(a) {
		return
	}
	if len(dst) > len(b) {
		return
	}
	for i := range dst {
		// amd64/v3:"VPADDD"
		// amd64/v4:"VPADDD"
		// arm64:"VADD V[0-9]+\\.S4"
		dst[i] = a[i] + b[i]
	}
}

//go:noinline
func reduction(x []uint32) uint32 {
	var sum uint32
	for _, v := range x {
		// amd64/v3:-"VPADDD"
		// amd64/v4:-"VPADDD"
		// arm64:-"VADD V[0-9]+\\.S4"
		sum += v
	}
	return sum
}

//go:noinline
func extraStores(x, y []uint32, k uint32) {
	if len(x) > len(y) {
		return
	}
	for i := range x {
		// amd64/v3:-"VPADDD"
		// amd64/v4:-"VPADDD"
		// arm64:-"VADD V[0-9]+\\.S4"
		x[i] += k
		y[i] += k
	}
}

//go:noinline
func unresolvedBounds(dst, src []uint32) {
	for i := range dst {
		// amd64/v3:-"VPADDD"
		// amd64/v4:-"VPADDD"
		// arm64:-"VADD V[0-9]+\\.S4"
		dst[i] += src[i]
	}
}

//go:noinline
func division(x []uint32, k uint32) {
	for i := range x {
		// amd64/v3:-"VPADDD"
		// amd64/v4:-"VPADDD"
		// arm64:-"VADD V[0-9]+\\.S4"
		x[i] = x[i]/k + 7
	}
}

func main() {}
