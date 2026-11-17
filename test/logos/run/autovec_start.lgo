// run -goexperiment simd

package main

//go:noinline
func startAdd(dst, src []uint32, start, end int) int {
	if start < 0 || end > len(dst) || end > len(src) {
		return start
	}
	i := start
	for ; i < end; i++ {
		dst[i] += src[i]
	}
	return i
}

//go:noinline
func startScalarAdd(a, b uint32) uint32 { return a + b }

func startReference(dst, src []uint32, start, end int) int {
	if start < 0 || end > len(dst) || end > len(src) {
		return start
	}
	i := start
	for ; i < end; i++ {
		dst[i] = startScalarAdd(dst[i], src[i])
	}
	return i
}

const startTestLength = 48

func checkStartAlias(start, end, distance int, reverse bool) {
	dstOffset, srcOffset := 1, 1+distance
	if reverse {
		dstOffset, srcOffset = srcOffset, dstOffset
	}
	backing := make([]uint32, startTestLength+distance+2)
	for i := range backing {
		backing[i] = uint32(i+1)*0x9e3779b9 + 0x80000001
	}
	want := append([]uint32(nil), backing...)
	dst := backing[dstOffset : dstOffset+startTestLength]
	src := backing[srcOffset : srcOffset+startTestLength]
	wantDst := want[dstOffset : dstOffset+startTestLength]
	wantSrc := want[srcOffset : srcOffset+startTestLength]
	wantIndex := startReference(wantDst, wantSrc, start, end)
	if got := startAdd(dst, src, start, end); got != wantIndex {
		panic("nonzero-start final index changed")
	}
	for i, got := range backing {
		if got != want[i] {
			panic("nonzero-start memory effects changed")
		}
	}
}

func checkStartInterval(start, end int) {
	for _, distance := range []int{0, 1, 2, 3, 4, 5, 7, 8, 12, 16, startTestLength} {
		for _, reverse := range []bool{false, true} {
			checkStartAlias(start, end, distance, reverse)
		}
	}
}

func main() {
	for _, start := range []int{-1, 0, 1, 2, 3, 4, 5, 8, 13, 31, 47, 48, 49} {
		for _, end := range []int{-1, 0, 1, 3, 4, 5, 8, 9, 12, 13, 16, 17, 31, 47, 48, 49} {
			checkStartInterval(start, end)
		}
	}
}
