// run -goexperiment simd

package main

import (
	"fmt"
	"math"
	"slices"
)

type integer interface {
	~int8 | ~uint8 | ~int16 | ~uint16 | ~int32 | ~uint32 | ~int64 | ~uint64
}

// scalarStep is deliberately a call in each reference iteration. A compiler
// bug that transforms the candidate must not silently transform its oracle.
//
//go:noinline
func scalarStep[T integer](v, k T, op byte) T {
	switch op {
	case '+':
		return v + k
	case '-':
		return v - k
	case '&':
		return v & k
	case '|':
		return v | k
	case '^':
		return v ^ k
	case '*':
		return v * k
	}
	panic("invalid scalar oracle operation")
}

//go:noinline
func addI8(x []int8, k int8) {
	for i := range x {
		x[i] += k
	}
}

//go:noinline
func addU8(x []uint8, k uint8) {
	for i := range x {
		x[i] += k
	}
}

//go:noinline
func addI16(x []int16, k int16) {
	for i := range x {
		x[i] += k
	}
}

//go:noinline
func addU16(x []uint16, k uint16) {
	for i := range x {
		x[i] += k
	}
}

//go:noinline
func addI32(x []int32, k int32) {
	for i := range x {
		x[i] += k
	}
}

//go:noinline
func addU32(x []uint32, k uint32) {
	for i := range x {
		x[i] += k
	}
}

//go:noinline
func addI64(x []int64, k int64) {
	for i := range x {
		x[i] += k
	}
}

//go:noinline
func addU64(x []uint64, k uint64) {
	for i := range x {
		x[i] += k
	}
}

//go:noinline
func subI32(x []int32, k int32) {
	for i := range x {
		x[i] -= k
	}
}

//go:noinline
func andU32(x []uint32, k uint32) {
	for i := range x {
		x[i] &= k
	}
}

//go:noinline
func orU32(x []uint32, k uint32) {
	for i := range x {
		x[i] |= k
	}
}

//go:noinline
func xorU32(x []uint32, k uint32) {
	for i := range x {
		x[i] ^= k
	}
}

//go:noinline
func mulI16(x []int16, k int16) {
	for i := range x {
		x[i] *= k
	}
}

//go:noinline
func mulU32(x []uint32, k uint32) {
	for i := range x {
		x[i] *= k
	}
}

func checkKernel[T integer](name string, lanes int, op byte, kernel func([]T, T)) {
	seed := uint64(0xfedcba9876543281)
	k := T(seed)
	kernel(nil, k)
	for _, n := range []int{0, 1, lanes - 1, lanes, lanes + 1, 2*lanes - 1, 2 * lanes, 2*lanes + 1, 7*lanes + 3} {
		backing := make([]T, n+2)
		backing[0], backing[n+1] = 11, 23
		x := backing[1 : n+1]
		want := make([]T, n)
		for i := range x {
			bits := uint64(i+1)*0x9e3779b97f4a7c15 ^ seed
			x[i] = T(bits)
			want[i] = scalarStep(x[i], k, op)
		}
		kernel(x, k)
		if !slices.Equal(x, want) || backing[0] != 11 || backing[n+1] != 23 {
			panic(fmt.Sprintf("%s length %d: got %v want %v; canaries %v/%v", name, n, x, want, backing[0], backing[n+1]))
		}
	}
}

//go:noinline
func addSlices(dst, src []int32) {
	if len(dst) > len(src) {
		return
	}
	for i := range dst {
		dst[i] += src[i]
	}
}

func checkOverlap() {
	addSlices(nil, nil)
	addSlices(nil, []int32{1})
	unchanged := []int32{3, 5, 7}
	addSlices(unchanged, nil)
	if !slices.Equal(unchanged, []int32{3, 5, 7}) {
		panic("short source changed destination despite early return")
	}
	for _, n := range []int{0, 1, 3, 4, 5, 7, 8, 9, 35} {
		for _, layout := range []struct {
			name string
			dst  int
			src  int
		}{
			{"same", 1, 1},
			{"forward", 2, 1},
			{"backward", 1, 2},
			{"disjoint", 1, n + 2},
		} {
			x := make([]int32, 2*n+4)
			for i := range x {
				x[i] = int32(uint32(i+1)*0x9e3779b9 + 0x7fffffff)
			}
			want := slices.Clone(x)
			for i := 0; i < n; i++ {
				want[layout.dst+i] = scalarStep(want[layout.dst+i], want[layout.src+i], '+')
			}
			addSlices(x[layout.dst:layout.dst+n], x[layout.src:layout.src+n])
			if !slices.Equal(x, want) {
				panic(fmt.Sprintf("overlap %s length %d: got %v want %v", layout.name, n, x, want))
			}
		}
	}
}

//go:noinline
func uncheckedAdd(dst, src []int32) {
	for i := range dst {
		dst[i] += src[i]
	}
}

func checkPanic() {
	for _, short := range []int{0, 1, 3, 4, 5, 8, 9} {
		dst := make([]int32, 13)
		src := make([]int32, short)
		for i := range src {
			src[i] = int32(i + 1)
		}
		panicked := false
		func() {
			defer func() { panicked = recover() != nil }()
			uncheckedAdd(dst, src)
		}()
		if !panicked {
			panic("short source did not panic")
		}
		for i, got := range dst {
			var want int32
			if i < short {
				want = int32(i + 1)
			}
			if got != want {
				panic(fmt.Sprintf("panic prefix length %d index %d: got %d want %d", short, i, got, want))
			}
		}
	}
}

//go:noinline
func floatSum(x []float64) float64 {
	var sum float64
	for _, v := range x {
		sum += v
	}
	return sum
}

//go:noinline
func scalarFloatAdd(a, b float64) float64 { return a + b }

func checkScalarSemantics() {
	for _, x := range [][]float64{
		{1e100, 1, -1e100, 1, 1},
		{math.Inf(1), math.Inf(-1), 1, 2, 3},
		{math.SmallestNonzeroFloat64, -math.SmallestNonzeroFloat64, math.Copysign(0, -1)},
	} {
		var want float64
		for _, v := range x {
			want = scalarFloatAdd(want, v)
		}
		got := floatSum(x)
		if math.IsNaN(want) {
			if !math.IsNaN(got) {
				panic("float reduction changed NaN")
			}
		} else if math.Float64bits(got) != math.Float64bits(want) {
			panic("float reduction reassociated")
		}
	}
	x := make([]int32, 37)
	for i := range x {
		x[i] = 1
	}
	for i := 1; i < len(x); i++ {
		x[i] += x[i-1]
	}
	for i, v := range x {
		if v != int32(i+1) {
			panic("cross-iteration dependency changed")
		}
	}
}

func main() {
	checkKernel("int8 add", 16, '+', addI8)
	checkKernel("uint8 add", 16, '+', addU8)
	checkKernel("int16 add", 8, '+', addI16)
	checkKernel("uint16 add", 8, '+', addU16)
	checkKernel("int32 add", 4, '+', addI32)
	checkKernel("uint32 add", 4, '+', addU32)
	checkKernel("int64 add", 2, '+', addI64)
	checkKernel("uint64 add", 2, '+', addU64)
	checkKernel("int32 subtract", 4, '-', subI32)
	checkKernel("uint32 and", 4, '&', andU32)
	checkKernel("uint32 or", 4, '|', orU32)
	checkKernel("uint32 xor", 4, '^', xorU32)
	checkKernel("int16 multiply", 8, '*', mulI16)
	checkKernel("uint32 multiply", 4, '*', mulU32)
	checkOverlap()
	checkPanic()
	checkScalarSemantics()
}
