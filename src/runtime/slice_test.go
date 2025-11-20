// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"fmt"
	"internal/asan"
	"internal/msan"
	"internal/race"
	"internal/testenv"
	"runtime"
	"testing"
)

const N = 20

func BenchmarkMakeSliceCopy(b *testing.B) {
	const length = 32
	var bytes = make([]byte, 8*length)
	var ints = make([]int, length)
	var ptrs = make([]*byte, length)
	b.Run("mallocmove", func(b *testing.B) {
		b.Run("Byte", func(b *testing.B) {
			var x []byte
			for i := 0; i < b.N; i++ {
				x = make([]byte, len(bytes))
				copy(x, bytes)
			}
		})
		b.Run("Int", func(b *testing.B) {
			var x []int
			for i := 0; i < b.N; i++ {
				x = make([]int, len(ints))
				copy(x, ints)
			}
		})
		b.Run("Ptr", func(b *testing.B) {
			var x []*byte
			for i := 0; i < b.N; i++ {
				x = make([]*byte, len(ptrs))
				copy(x, ptrs)
			}

		})
	})
	b.Run("makecopy", func(b *testing.B) {
		b.Run("Byte", func(b *testing.B) {
			var x []byte
			for i := 0; i < b.N; i++ {
				x = make([]byte, 8*length)
				copy(x, bytes)
			}
		})
		b.Run("Int", func(b *testing.B) {
			var x []int
			for i := 0; i < b.N; i++ {
				x = make([]int, length)
				copy(x, ints)
			}
		})
		b.Run("Ptr", func(b *testing.B) {
			var x []*byte
			for i := 0; i < b.N; i++ {
				x = make([]*byte, length)
				copy(x, ptrs)
			}

		})
	})
	b.Run("nilappend", func(b *testing.B) {
		b.Run("Byte", func(b *testing.B) {
			var x []byte
			for i := 0; i < b.N; i++ {
				x = append([]byte(nil), bytes...)
				_ = x
			}
		})
		b.Run("Int", func(b *testing.B) {
			var x []int
			for i := 0; i < b.N; i++ {
				x = append([]int(nil), ints...)
				_ = x
			}
		})
		b.Run("Ptr", func(b *testing.B) {
			var x []*byte
			for i := 0; i < b.N; i++ {
				x = append([]*byte(nil), ptrs...)
				_ = x
			}
		})
	})
}

type (
	struct24 struct{ a, b, c int64 }
	struct32 struct{ a, b, c, d int64 }
	struct40 struct{ a, b, c, d, e int64 }
)

func BenchmarkMakeSlice(b *testing.B) {
	const length = 2
	b.Run("Byte", func(b *testing.B) {
		var x []byte
		for i := 0; i < b.N; i++ {
			x = make([]byte, length, 2*length)
			_ = x
		}
	})
	b.Run("Int16", func(b *testing.B) {
		var x []int16
		for i := 0; i < b.N; i++ {
			x = make([]int16, length, 2*length)
			_ = x
		}
	})
	b.Run("Int", func(b *testing.B) {
		var x []int
		for i := 0; i < b.N; i++ {
			x = make([]int, length, 2*length)
			_ = x
		}
	})
	b.Run("Ptr", func(b *testing.B) {
		var x []*byte
		for i := 0; i < b.N; i++ {
			x = make([]*byte, length, 2*length)
			_ = x
		}
	})
	b.Run("Struct", func(b *testing.B) {
		b.Run("24", func(b *testing.B) {
			var x []struct24
			for i := 0; i < b.N; i++ {
				x = make([]struct24, length, 2*length)
				_ = x
			}
		})
		b.Run("32", func(b *testing.B) {
			var x []struct32
			for i := 0; i < b.N; i++ {
				x = make([]struct32, length, 2*length)
				_ = x
			}
		})
		b.Run("40", func(b *testing.B) {
			var x []struct40
			for i := 0; i < b.N; i++ {
				x = make([]struct40, length, 2*length)
				_ = x
			}
		})

	})
}

func BenchmarkGrowSlice(b *testing.B) {
	b.Run("Byte", func(b *testing.B) {
		x := make([]byte, 9)
		for i := 0; i < b.N; i++ {
			_ = append([]byte(nil), x...)
		}
	})
	b.Run("Int16", func(b *testing.B) {
		x := make([]int16, 9)
		for i := 0; i < b.N; i++ {
			_ = append([]int16(nil), x...)
		}
	})
	b.Run("Int", func(b *testing.B) {
		x := make([]int, 9)
		for i := 0; i < b.N; i++ {
			_ = append([]int(nil), x...)
		}
	})
	b.Run("Ptr", func(b *testing.B) {
		x := make([]*byte, 9)
		for i := 0; i < b.N; i++ {
			_ = append([]*byte(nil), x...)
		}
	})
	b.Run("Struct", func(b *testing.B) {
		b.Run("24", func(b *testing.B) {
			x := make([]struct24, 9)
			for i := 0; i < b.N; i++ {
				_ = append([]struct24(nil), x...)
			}
		})
		b.Run("32", func(b *testing.B) {
			x := make([]struct32, 9)
			for i := 0; i < b.N; i++ {
				_ = append([]struct32(nil), x...)
			}
		})
		b.Run("40", func(b *testing.B) {
			x := make([]struct40, 9)
			for i := 0; i < b.N; i++ {
				_ = append([]struct40(nil), x...)
			}
		})

	})
}

var (
	SinkIntSlice        []int
	SinkIntPointerSlice []*int
)

func BenchmarkExtendSlice(b *testing.B) {
	var length = 4 // Use a variable to prevent stack allocation of slices.
	b.Run("IntSlice", func(b *testing.B) {
		s := make([]int, 0, length)
		for i := 0; i < b.N; i++ {
			s = append(s[:0:length/2], make([]int, length)...)
		}
		SinkIntSlice = s
	})
	b.Run("PointerSlice", func(b *testing.B) {
		s := make([]*int, 0, length)
		for i := 0; i < b.N; i++ {
			s = append(s[:0:length/2], make([]*int, length)...)
		}
		SinkIntPointerSlice = s
	})
	b.Run("NoGrow", func(b *testing.B) {
		s := make([]int, 0, length)
		for i := 0; i < b.N; i++ {
			s = append(s[:0:length], make([]int, length)...)
		}
		SinkIntSlice = s
	})
}

func BenchmarkAppend(b *testing.B) {
	b.StopTimer()
	x := make([]int, 0, N)
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		x = x[0:0]
		for j := 0; j < N; j++ {
			x = append(x, j)
		}
	}
}

func BenchmarkAppendGrowByte(b *testing.B) {
	for i := 0; i < b.N; i++ {
		var x []byte
		for j := 0; j < 1<<20; j++ {
			x = append(x, byte(j))
		}
	}
}

func BenchmarkAppendGrowString(b *testing.B) {
	var s string
	for i := 0; i < b.N; i++ {
		var x []string
		for j := 0; j < 1<<20; j++ {
			x = append(x, s)
		}
	}
}

func BenchmarkAppendSlice(b *testing.B) {
	for _, length := range []int{1, 4, 7, 8, 15, 16, 32} {
		b.Run(fmt.Sprint(length, "Bytes"), func(b *testing.B) {
			x := make([]byte, 0, N)
			y := make([]byte, length)
			for i := 0; i < b.N; i++ {
				x = x[0:0]
				x = append(x, y...)
			}
		})
	}
}

var (
	blackhole []byte
)

func BenchmarkAppendSliceLarge(b *testing.B) {
	for _, length := range []int{1 << 10, 4 << 10, 16 << 10, 64 << 10, 256 << 10, 1024 << 10} {
		y := make([]byte, length)
		b.Run(fmt.Sprint(length, "Bytes"), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				blackhole = nil
				blackhole = append(blackhole, y...)
			}
		})
	}
}

func BenchmarkAppendStr(b *testing.B) {
	for _, str := range []string{
		"1",
		"1234",
		"12345678",
		"1234567890123456",
		"12345678901234567890123456789012",
	} {
		b.Run(fmt.Sprint(len(str), "Bytes"), func(b *testing.B) {
			x := make([]byte, 0, N)
			for i := 0; i < b.N; i++ {
				x = x[0:0]
				x = append(x, str...)
			}
		})
	}
}

func BenchmarkAppendSpecialCase(b *testing.B) {
	b.StopTimer()
	x := make([]int, 0, N)
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		x = x[0:0]
		for j := 0; j < N; j++ {
			if len(x) < cap(x) {
				x = x[:len(x)+1]
				x[len(x)-1] = j
			} else {
				x = append(x, j)
			}
		}
	}
}

var x []int

func f() int {
	x[:1][0] = 3
	return 2
}

func TestSideEffectOrder(t *testing.T) {
	x = make([]int, 0, 10)
	x = append(x, 1, f())
	if x[0] != 1 || x[1] != 2 {
		t.Error("append failed: ", x[0], x[1])
	}
}

func TestAppendOverlap(t *testing.T) {
	x := []byte("1234")
	x = append(x[1:], x...) // p > q in runtime·appendslice.
	got := string(x)
	want := "2341234"
	if got != want {
		t.Errorf("overlap failed: got %q want %q", got, want)
	}
}

func BenchmarkCopy(b *testing.B) {
	for _, l := range []int{1, 2, 4, 8, 12, 16, 32, 128, 1024} {
		buf := make([]byte, 4096)
		b.Run(fmt.Sprint(l, "Byte"), func(b *testing.B) {
			s := make([]byte, l)
			var n int
			for i := 0; i < b.N; i++ {
				n = copy(buf, s)
			}
			b.SetBytes(int64(n))
		})
		b.Run(fmt.Sprint(l, "String"), func(b *testing.B) {
			s := string(make([]byte, l))
			var n int
			for i := 0; i < b.N; i++ {
				n = copy(buf, s)
			}
			b.SetBytes(int64(n))
		})
	}
}

var (
	sByte []byte
	s1Ptr []uintptr
	s2Ptr [][2]uintptr
	s3Ptr [][3]uintptr
	s4Ptr [][4]uintptr
)

// BenchmarkAppendInPlace tests the performance of append
// when the result is being written back to the same slice.
// In order for the in-place optimization to occur,
// the slice must be referred to by address;
// using a global is an easy way to trigger that.
// We test the "grow" and "no grow" paths separately,
// but not the "normal" (occasionally grow) path,
// because it is a blend of the other two.
// We use small numbers and small sizes in an attempt
// to avoid benchmarking memory allocation and copying.
// We use scalars instead of pointers in an attempt
// to avoid benchmarking the write barriers.
// We benchmark four common sizes (byte, pointer, string/interface, slice),
// and one larger size.
func BenchmarkAppendInPlace(b *testing.B) {
	b.Run("NoGrow", func(b *testing.B) {
		const C = 128

		b.Run("Byte", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				sByte = make([]byte, C)
				for j := 0; j < C; j++ {
					sByte = append(sByte, 0x77)
				}
			}
		})

		b.Run("1Ptr", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				s1Ptr = make([]uintptr, C)
				for j := 0; j < C; j++ {
					s1Ptr = append(s1Ptr, 0x77)
				}
			}
		})

		b.Run("2Ptr", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				s2Ptr = make([][2]uintptr, C)
				for j := 0; j < C; j++ {
					s2Ptr = append(s2Ptr, [2]uintptr{0x77, 0x88})
				}
			}
		})

		b.Run("3Ptr", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				s3Ptr = make([][3]uintptr, C)
				for j := 0; j < C; j++ {
					s3Ptr = append(s3Ptr, [3]uintptr{0x77, 0x88, 0x99})
				}
			}
		})

		b.Run("4Ptr", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				s4Ptr = make([][4]uintptr, C)
				for j := 0; j < C; j++ {
					s4Ptr = append(s4Ptr, [4]uintptr{0x77, 0x88, 0x99, 0xAA})
				}
			}
		})

	})

	b.Run("Grow", func(b *testing.B) {
		const C = 5

		b.Run("Byte", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				sByte = make([]byte, 0)
				for j := 0; j < C; j++ {
					sByte = append(sByte, 0x77)
					sByte = sByte[:cap(sByte)]
				}
			}
		})

		b.Run("1Ptr", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				s1Ptr = make([]uintptr, 0)
				for j := 0; j < C; j++ {
					s1Ptr = append(s1Ptr, 0x77)
					s1Ptr = s1Ptr[:cap(s1Ptr)]
				}
			}
		})

		b.Run("2Ptr", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				s2Ptr = make([][2]uintptr, 0)
				for j := 0; j < C; j++ {
					s2Ptr = append(s2Ptr, [2]uintptr{0x77, 0x88})
					s2Ptr = s2Ptr[:cap(s2Ptr)]
				}
			}
		})

		b.Run("3Ptr", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				s3Ptr = make([][3]uintptr, 0)
				for j := 0; j < C; j++ {
					s3Ptr = append(s3Ptr, [3]uintptr{0x77, 0x88, 0x99})
					s3Ptr = s3Ptr[:cap(s3Ptr)]
				}
			}
		})

		b.Run("4Ptr", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				s4Ptr = make([][4]uintptr, 0)
				for j := 0; j < C; j++ {
					s4Ptr = append(s4Ptr, [4]uintptr{0x77, 0x88, 0x99, 0xAA})
					s4Ptr = s4Ptr[:cap(s4Ptr)]
				}
			}
		})

	})
}

//go:noinline
func byteSlice(n int) []byte {
	var r []byte
	for i := range n {
		r = append(r, byte(i))
	}
	return r
}
func TestAppendByteInLoop(t *testing.T) {
	testenv.SkipIfOptimizationOff(t)
	if race.Enabled {
		t.Skip("skipping in -race mode")
	}
	if asan.Enabled || msan.Enabled {
		t.Skip("skipping in sanitizer mode")
	}
	for _, test := range [][3]int{
		{0, 0, 0},
		{1, 1, 8},
		{2, 1, 8},
		{8, 1, 8},
		{9, 1, 16},
		{16, 1, 16},
		{17, 1, 24},
		{24, 1, 24},
		{25, 1, 32},
		{32, 1, 32},
		{33, 1, 64}, // If we up the stack buffer size from 32->64, this line and the next would become 48.
		{48, 1, 64},
		{49, 1, 64},
		{64, 1, 64},
		{65, 2, 128},
	} {
		n := test[0]
		want := test[1]
		wantCap := test[2]
		var r []byte
		got := testing.AllocsPerRun(10, func() {
			r = byteSlice(n)
		})
		if got != float64(want) {
			t.Errorf("for size %d, got %f allocs want %d", n, got, want)
		}
		if cap(r) != wantCap {
			t.Errorf("for size %d, got capacity %d want %d", n, cap(r), wantCap)
		}
	}
}

//go:noinline
func ptrSlice(n int, p *[]*byte) {
	var r []*byte
	for range n {
		r = append(r, nil)
	}
	*p = r
}
func TestAppendPtrInLoop(t *testing.T) {
	testenv.SkipIfOptimizationOff(t)
	if race.Enabled {
		t.Skip("skipping in -race mode")
	}
	if asan.Enabled || msan.Enabled {
		t.Skip("skipping in sanitizer mode")
	}
	var tests [][3]int
	if runtime.PtrSize == 8 {
		tests = [][3]int{
			{0, 0, 0},
			{1, 1, 1},
			{2, 1, 2},
			{3, 1, 3}, // This is the interesting case, allocates 24 bytes when before it was 32.
			{4, 1, 4},
			{5, 1, 8},
			{6, 1, 8},
			{7, 1, 8},
			{8, 1, 8},
			{9, 2, 16},
		}
	} else {
		tests = [][3]int{
			{0, 0, 0},
			{1, 1, 2},
			{2, 1, 2},
			{3, 1, 4},
			{4, 1, 4},
			{5, 1, 6}, // These two are also 24 bytes instead of 32.
			{6, 1, 6}, //
			{7, 1, 8},
			{8, 1, 8},
			{9, 1, 16},
			{10, 1, 16},
			{11, 1, 16},
			{12, 1, 16},
			{13, 1, 16},
			{14, 1, 16},
			{15, 1, 16},
			{16, 1, 16},
			{17, 2, 32},
		}
	}
	for _, test := range tests {
		n := test[0]
		want := test[1]
		wantCap := test[2]
		var r []*byte
		got := testing.AllocsPerRun(10, func() {
			ptrSlice(n, &r)
		})
		if got != float64(want) {
			t.Errorf("for size %d, got %f allocs want %d", n, got, want)
		}
		if cap(r) != wantCap {
			t.Errorf("for size %d, got capacity %d want %d", n, cap(r), wantCap)
		}
	}
}

//go:noinline
func byteCapSlice(n int) ([]byte, int) {
	var r []byte
	for i := range n {
		r = append(r, byte(i))
	}
	return r, cap(r)
}
func TestAppendByteCapInLoop(t *testing.T) {
	testenv.SkipIfOptimizationOff(t)
	if race.Enabled {
		t.Skip("skipping in -race mode")
	}
	if asan.Enabled || msan.Enabled {
		t.Skip("skipping in sanitizer mode")
	}
	for _, test := range [][3]int{
		{0, 0, 0},
		{1, 1, 8},
		{2, 1, 8},
		{8, 1, 8},
		{9, 1, 16},
		{16, 1, 16},
		{17, 1, 24},
		{24, 1, 24},
		{25, 1, 32},
		{32, 1, 32},
		{33, 1, 64},
		{48, 1, 64},
		{49, 1, 64},
		{64, 1, 64},
		{65, 2, 128},
	} {
		n := test[0]
		want := test[1]
		wantCap := test[2]
		var r []byte
		got := testing.AllocsPerRun(10, func() {
			r, _ = byteCapSlice(n)
		})
		if got != float64(want) {
			t.Errorf("for size %d, got %f allocs want %d", n, got, want)
		}
		if cap(r) != wantCap {
			t.Errorf("for size %d, got capacity %d want %d", n, cap(r), wantCap)
		}
	}
}

func TestAppendGeneric(t *testing.T) {
	type I *int
	r := testAppendGeneric[I](100)
	if len(r) != 100 {
		t.Errorf("bad length")
	}
}

//go:noinline
func testAppendGeneric[E any](n int) []E {
	var r []E
	var z E
	for range n {
		r = append(r, z)
	}
	return r
}

func appendSomeBytes(r []byte, s []byte) []byte {
	for _, b := range s {
		r = append(r, b)
	}
	return r
}

func TestAppendOfArg(t *testing.T) {
	r := make([]byte, 24)
	for i := 0; i < 24; i++ {
		r[i] = byte(i)
	}
	appendSomeBytes(r, []byte{25, 26, 27})
	// Do the same thing, trying to overwrite any
	// stack-allocated buffers used above.
	s := make([]byte, 24)
	for i := 0; i < 24; i++ {
		s[i] = 99
	}
	appendSomeBytes(s, []byte{99, 99, 99})
	// Check that we still have the right data.
	for i, b := range r {
		if b != byte(i) {
			t.Errorf("r[%d]=%d, want %d", i, b, byte(i))
		}
	}

}

func BenchmarkAppendInLoop(b *testing.B) {
	for _, size := range []int{0, 1, 8, 16, 32, 64, 128} {
		b.Run(fmt.Sprintf("%d", size),
			func(b *testing.B) {
				b.ReportAllocs()
				for b.Loop() {
					byteSlice(size)
				}
			})
	}
}

func TestMoveToHeapEarly(t *testing.T) {
	// Just checking that this compiles.
	var x []int
	y := x // causes a move2heap in the entry block
	for range 5 {
		x = append(x, 5)
	}
	_ = y
}

func TestMoveToHeapCap(t *testing.T) {
	var c int
	r := func() []byte {
		var s []byte
		for i := range 10 {
			s = append(s, byte(i))
		}
		c = cap(s)
		return s
	}()
	if c != cap(r) {
		t.Errorf("got cap=%d, want %d", c, cap(r))
	}
	sinkSlice = r
}

//go:noinline
func runit(f func()) {
	f()
}

func TestMoveToHeapClosure1(t *testing.T) {
	var c int
	r := func() []byte {
		var s []byte
		for i := range 10 {
			s = append(s, byte(i))
		}
		runit(func() {
			c = cap(s)
		})
		return s
	}()
	if c != cap(r) {
		t.Errorf("got cap=%d, want %d", c, cap(r))
	}
	sinkSlice = r
}
func TestMoveToHeapClosure2(t *testing.T) {
	var c int
	r := func() []byte {
		var s []byte
		for i := range 10 {
			s = append(s, byte(i))
		}
		c = func() int {
			return cap(s)
		}()
		return s
	}()
	if c != cap(r) {
		t.Errorf("got cap=%d, want %d", c, cap(r))
	}
	sinkSlice = r
}

//go:noinline
func buildClosure(t *testing.T) ([]byte, func()) {
	var s []byte
	for i := range 20 {
		s = append(s, byte(i))
	}
	c := func() {
		for i, b := range s {
			if b != byte(i) {
				t.Errorf("s[%d]=%d, want %d", i, b, i)
			}
		}
	}
	return s, c
}

func TestMoveToHeapClosure3(t *testing.T) {
	_, f := buildClosure(t)
	overwriteStack(0)
	f()
}

//go:noinline
func overwriteStack(n int) uint64 {
	var x [100]uint64
	for i := range x {
		x[i] = 0xabcdabcdabcdabcd
	}
	return x[n]
}

var sinkSlice []byte
