// run -goexperiment simd

package main

type word uint32

//go:noinline
func scalarAdd(a, b uint32) uint32 { return a + b }

//go:noinline
func scalarXor(a, b uint32) uint32 { return a ^ b }

//go:noinline
func updateNamed(x []word, k word) {
	for i := range x {
		x[i] += k
	}
}

// updateBounded deliberately permits a negative bound. Its loop then executes
// zero iterations, and its externally visible final index must remain zero.
//
//go:noinline
func updateBounded(x []uint32, n int, k uint32) int {
	if n > len(x) {
		return -1
	}
	i := 0
	for ; i < n; i++ {
		x[i] += k
	}
	return i
}

//go:noinline
func sequential(x []uint32, add, xor uint32) {
	for i := range x {
		x[i] += add
	}
	for i := range x {
		x[i] ^= xor
	}
}

//go:noinline
func combine(dst, a, b []uint32) {
	if len(dst) > len(a) {
		return
	}
	if len(dst) > len(b) {
		return
	}
	for i := range dst {
		dst[i] = a[i] + b[i]
	}
}

func clone(x []uint32) []uint32 {
	out := make([]uint32, len(x))
	copy(out, x)
	return out
}

func equal(x, want []uint32) bool {
	if len(x) != len(want) {
		return false
	}
	for i, v := range x {
		if v != want[i] {
			return false
		}
	}
	return true
}

func input(n int) []uint32 {
	x := make([]uint32, n)
	for i := range x {
		x[i] = uint32(i+1)*0x9e3779b9 + 0x7fffffff
	}
	return x
}

func checkNamedAndSequential() {
	updateNamed(nil, 0xffffffff)
	sequential(nil, 0xffffffff, 0x76543210)
	for _, n := range []int{0, 1, 3, 4, 5, 8, 9, 33} {
		x := input(n)
		want := clone(x)
		named := make([]word, n)
		for i, v := range x {
			named[i] = word(v)
			want[i] = scalarAdd(v, 0xffffffff)
		}
		updateNamed(named, 0xffffffff)
		for i, v := range named {
			if uint32(v) != want[i] {
				panic("defined integer type changed arithmetic")
			}
		}
		sequential(x, 0xffffffff, 0x76543210)
		for i := range want {
			want[i] = scalarXor(want[i], 0x76543210)
		}
		if !equal(x, want) {
			panic("sequential vector loops changed result")
		}
	}
}

func checkBounded() {
	minInt := -int(^uint(0)>>1) - 1
	for _, n := range []int{minInt, -5, -1, 0, 1, 3, 4, 5, 8, 9, 16, 17, 18} {
		x := input(17)
		want := clone(x)
		wantIndex := n
		if n < 0 {
			wantIndex = 0
		} else if n > len(x) {
			wantIndex = -1
		} else {
			for i := 0; i < n; i++ {
				want[i] = scalarAdd(want[i], 0x80000001)
			}
		}
		if gotIndex := updateBounded(x, n, 0x80000001); gotIndex != wantIndex || !equal(x, want) {
			panic("bounded loop changed final index or written prefix")
		}
	}
	if updateBounded(nil, -1, 1) != 0 || updateBounded(nil, 0, 1) != 0 {
		panic("empty bounded loop changed final index")
	}
}

func checkThreeSlices() {
	combine(nil, nil, nil)
	for _, n := range []int{0, 1, 3, 4, 5, 8, 9, 35} {
		for _, layout := range []struct{ dst, a, b int }{
			{1, n + 2, 2*n + 3}, // disjoint
			{1, 1, n + 2},       // destination equals first source
			{1, n + 2, 1},       // destination equals second source
			{1, 1, 1},           // all equal
			{1, n + 2, n + 2},   // sources equal, destination disjoint
			{1, n + 2, n + 3},   // sources overlap, destination disjoint
			{2, 1, n + 3},       // first source overlaps forward
			{1, 2, n + 3},       // first source overlaps backward
			{2, n + 3, 1},       // second source overlaps forward
			{1, n + 3, 2},       // second source overlaps backward
		} {
			x := input(3*n + 5)
			want := clone(x)
			for i := 0; i < n; i++ {
				want[layout.dst+i] = scalarAdd(want[layout.a+i], want[layout.b+i])
			}
			combine(x[layout.dst:layout.dst+n], x[layout.a:layout.a+n], x[layout.b:layout.b+n])
			if !equal(x, want) {
				panic("three-slice aliasing changed scalar order")
			}
		}
	}
	// Each source guard must protect the whole operation before any store.
	for _, shortFirst := range []bool{false, true} {
		x := input(9)
		want := clone(x)
		a, b := input(9), input(9)
		if shortFirst {
			a = a[:8]
		} else {
			b = b[:8]
		}
		combine(x, a, b)
		if !equal(x, want) {
			panic("three-slice short source changed destination")
		}
	}
}

func checkAliasSpanEdges() {
	// A uint32 vector holds four lanes. At n=5 the vector prefix spans
	// four elements; distance three carries a dependency inside that vector.
	for _, n := range []int{5, 9, 17} {
		span := n / 4 * 4
		for _, distance := range []int{span - 1, span, span + 1} {
			for _, forward := range []bool{false, true} {
				dst, a := 1, 1+distance
				if forward {
					dst, a = a, dst
				}
				b := 2*n + distance + 3
				x := input(b + n + 1)
				want := clone(x)
				for i := 0; i < n; i++ {
					want[dst+i] = scalarAdd(want[a+i], want[b+i])
				}
				combine(x[dst:dst+n], x[a:a+n], x[b:b+n])
				if !equal(x, want) {
					panic("alias distance at vector-prefix span changed result")
				}
			}
		}
	}
}

func main() {
	checkNamedAndSequential()
	checkBounded()
	checkThreeSlices()
	checkAliasSpanEdges()
}
