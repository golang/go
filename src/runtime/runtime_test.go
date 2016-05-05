// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"io"
	. "runtime"
	"runtime/debug"
	"testing"
	"unsafe"
)

func init() {
	// We're testing the runtime, so make tracebacks show things
	// in the runtime. This only raises the level, so it won't
	// override GOTRACEBACK=crash from the user.
	SetTracebackEnv("system")
}

var errf error

func errfn() error {
	return errf
}

func errfn1() error {
	return io.EOF
}

func BenchmarkIfaceCmp100(b *testing.B) {
	for i := 0; i < b.N; i++ {
		for j := 0; j < 100; j++ {
			if errfn() == io.EOF {
				b.Fatal("bad comparison")
			}
		}
	}
}

func BenchmarkIfaceCmpNil100(b *testing.B) {
	for i := 0; i < b.N; i++ {
		for j := 0; j < 100; j++ {
			if errfn1() == nil {
				b.Fatal("bad comparison")
			}
		}
	}
}

func BenchmarkDefer(b *testing.B) {
	for i := 0; i < b.N; i++ {
		defer1()
	}
}

func defer1() {
	defer func(x, y, z int) {
		if recover() != nil || x != 1 || y != 2 || z != 3 {
			panic("bad recover")
		}
	}(1, 2, 3)
	return
}

func BenchmarkDefer10(b *testing.B) {
	for i := 0; i < b.N/10; i++ {
		defer2()
	}
}

func defer2() {
	for i := 0; i < 10; i++ {
		defer func(x, y, z int) {
			if recover() != nil || x != 1 || y != 2 || z != 3 {
				panic("bad recover")
			}
		}(1, 2, 3)
	}
}

func BenchmarkDeferMany(b *testing.B) {
	for i := 0; i < b.N; i++ {
		defer func(x, y, z int) {
			if recover() != nil || x != 1 || y != 2 || z != 3 {
				panic("bad recover")
			}
		}(1, 2, 3)
	}
}

// golang.org/issue/7063
func TestStopCPUProfilingWithProfilerOff(t *testing.T) {
	SetCPUProfileRate(0)
}

// Addresses to test for faulting behavior.
// This is less a test of SetPanicOnFault and more a check that
// the operating system and the runtime can process these faults
// correctly. That is, we're indirectly testing that without SetPanicOnFault
// these would manage to turn into ordinary crashes.
// Note that these are truncated on 32-bit systems, so the bottom 32 bits
// of the larger addresses must themselves be invalid addresses.
// We might get unlucky and the OS might have mapped one of these
// addresses, but probably not: they're all in the first page, very high
// addresses that normally an OS would reserve for itself, or malformed
// addresses. Even so, we might have to remove one or two on different
// systems. We will see.

var faultAddrs = []uint64{
	// low addresses
	0,
	1,
	0xfff,
	// high (kernel) addresses
	// or else malformed.
	0xffffffffffffffff,
	0xfffffffffffff001,
	0xffffffffffff0001,
	0xfffffffffff00001,
	0xffffffffff000001,
	0xfffffffff0000001,
	0xffffffff00000001,
	0xfffffff000000001,
	0xffffff0000000001,
	0xfffff00000000001,
	0xffff000000000001,
	0xfff0000000000001,
	0xff00000000000001,
	0xf000000000000001,
	0x8000000000000001,
}

func TestSetPanicOnFault(t *testing.T) {
	old := debug.SetPanicOnFault(true)
	defer debug.SetPanicOnFault(old)

	nfault := 0
	for _, addr := range faultAddrs {
		testSetPanicOnFault(t, uintptr(addr), &nfault)
	}
	if nfault == 0 {
		t.Fatalf("none of the addresses faulted")
	}
}

func testSetPanicOnFault(t *testing.T, addr uintptr, nfault *int) {
	if GOOS == "nacl" {
		t.Skip("nacl doesn't seem to fault on high addresses")
	}

	defer func() {
		if err := recover(); err != nil {
			*nfault++
		}
	}()

	// The read should fault, except that sometimes we hit
	// addresses that have had C or kernel pages mapped there
	// readable by user code. So just log the content.
	// If no addresses fault, we'll fail the test.
	v := *(*byte)(unsafe.Pointer(addr))
	t.Logf("addr %#x: %#x\n", addr, v)
}

func eqstring_generic(s1, s2 string) bool {
	if len(s1) != len(s2) {
		return false
	}
	// optimization in assembly versions:
	// if s1.str == s2.str { return true }
	for i := 0; i < len(s1); i++ {
		if s1[i] != s2[i] {
			return false
		}
	}
	return true
}

func TestEqString(t *testing.T) {
	// This isn't really an exhaustive test of eqstring, it's
	// just a convenient way of documenting (via eqstring_generic)
	// what eqstring does.
	s := []string{
		"",
		"a",
		"c",
		"aaa",
		"ccc",
		"cccc"[:3], // same contents, different string
		"1234567890",
	}
	for _, s1 := range s {
		for _, s2 := range s {
			x := s1 == s2
			y := eqstring_generic(s1, s2)
			if x != y {
				t.Errorf(`eqstring("%s","%s") = %t, want %t`, s1, s2, x, y)
			}
		}
	}
}

func TestTrailingZero(t *testing.T) {
	// make sure we add padding for structs with trailing zero-sized fields
	type T1 struct {
		n int32
		z [0]byte
	}
	if unsafe.Sizeof(T1{}) != 8 {
		t.Errorf("sizeof(%#v)==%d, want 8", T1{}, unsafe.Sizeof(T1{}))
	}
	type T2 struct {
		n int64
		z struct{}
	}
	if unsafe.Sizeof(T2{}) != 8+unsafe.Sizeof(Uintreg(0)) {
		t.Errorf("sizeof(%#v)==%d, want %d", T2{}, unsafe.Sizeof(T2{}), 8+unsafe.Sizeof(Uintreg(0)))
	}
	type T3 struct {
		n byte
		z [4]struct{}
	}
	if unsafe.Sizeof(T3{}) != 2 {
		t.Errorf("sizeof(%#v)==%d, want 2", T3{}, unsafe.Sizeof(T3{}))
	}
	// make sure padding can double for both zerosize and alignment
	type T4 struct {
		a int32
		b int16
		c int8
		z struct{}
	}
	if unsafe.Sizeof(T4{}) != 8 {
		t.Errorf("sizeof(%#v)==%d, want 8", T4{}, unsafe.Sizeof(T4{}))
	}
	// make sure we don't pad a zero-sized thing
	type T5 struct {
	}
	if unsafe.Sizeof(T5{}) != 0 {
		t.Errorf("sizeof(%#v)==%d, want 0", T5{}, unsafe.Sizeof(T5{}))
	}
}

func TestBadOpen(t *testing.T) {
	if GOOS == "windows" || GOOS == "nacl" {
		t.Skip("skipping OS that doesn't have open/read/write/close")
	}
	// make sure we get the correct error code if open fails. Same for
	// read/write/close on the resulting -1 fd. See issue 10052.
	nonfile := []byte("/notreallyafile")
	fd := Open(&nonfile[0], 0, 0)
	if fd != -1 {
		t.Errorf("open(\"%s\")=%d, want -1", string(nonfile), fd)
	}
	var buf [32]byte
	r := Read(-1, unsafe.Pointer(&buf[0]), int32(len(buf)))
	if r != -1 {
		t.Errorf("read()=%d, want -1", r)
	}
	w := Write(^uintptr(0), unsafe.Pointer(&buf[0]), int32(len(buf)))
	if w != -1 {
		t.Errorf("write()=%d, want -1", w)
	}
	c := Close(-1)
	if c != -1 {
		t.Errorf("close()=%d, want -1", c)
	}
}

func TestAppendGrowth(t *testing.T) {
	var x []int64
	check := func(want int) {
		if cap(x) != want {
			t.Errorf("len=%d, cap=%d, want cap=%d", len(x), cap(x), want)
		}
	}

	check(0)
	want := 1
	for i := 1; i <= 100; i++ {
		x = append(x, 1)
		check(want)
		if i&(i-1) == 0 {
			want = 2 * i
		}
	}
}

var One = []int64{1}

func TestAppendSliceGrowth(t *testing.T) {
	var x []int64
	check := func(want int) {
		if cap(x) != want {
			t.Errorf("len=%d, cap=%d, want cap=%d", len(x), cap(x), want)
		}
	}

	check(0)
	want := 1
	for i := 1; i <= 100; i++ {
		x = append(x, One...)
		check(want)
		if i&(i-1) == 0 {
			want = 2 * i
		}
	}
}

func TestGoroutineProfileTrivial(t *testing.T) {
	// Calling GoroutineProfile twice in a row should find the same number of goroutines,
	// but it's possible there are goroutines just about to exit, so we might end up
	// with fewer in the second call. Try a few times; it should converge once those
	// zombies are gone.
	for i := 0; ; i++ {
		n1, ok := GoroutineProfile(nil) // should fail, there's at least 1 goroutine
		if n1 < 1 || ok {
			t.Fatalf("GoroutineProfile(nil) = %d, %v, want >0, false", n1, ok)
		}
		n2, ok := GoroutineProfile(make([]StackRecord, n1))
		if n2 == n1 && ok {
			break
		}
		t.Logf("GoroutineProfile(%d) = %d, %v, want %d, true", n1, n2, ok, n1)
		if i >= 10 {
			t.Fatalf("GoroutineProfile not converging")
		}
	}
}
