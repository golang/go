// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"io"
	"io/ioutil"
	"os"
	"os/exec"
	. "runtime"
	"runtime/debug"
	"strconv"
	"strings"
	"testing"
	"unsafe"
)

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

// The profiling signal handler needs to know whether it is executing runtime.gogo.
// The constant RuntimeGogoBytes in arch_*.h gives the size of the function;
// we don't have a way to obtain it from the linker (perhaps someday).
// Test that the constant matches the size determined by 'go tool nm -S'.
// The value reported will include the padding between runtime.gogo and the
// next function in memory. That's fine.
func TestRuntimeGogoBytes(t *testing.T) {
	switch GOOS {
	case "android", "nacl":
		t.Skipf("skipping on %s", GOOS)
	}

	dir, err := ioutil.TempDir("", "go-build")
	if err != nil {
		t.Fatalf("failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(dir)

	out, err := exec.Command("go", "build", "-o", dir+"/hello", "../../test/helloworld.go").CombinedOutput()
	if err != nil {
		t.Fatalf("building hello world: %v\n%s", err, out)
	}

	out, err = exec.Command("go", "tool", "nm", "-size", dir+"/hello").CombinedOutput()
	if err != nil {
		t.Fatalf("go tool nm: %v\n%s", err, out)
	}

	for _, line := range strings.Split(string(out), "\n") {
		f := strings.Fields(line)
		if len(f) == 4 && f[3] == "runtime.gogo" {
			size, _ := strconv.Atoi(f[1])
			if GogoBytes() != int32(size) {
				t.Fatalf("RuntimeGogoBytes = %d, should be %d", GogoBytes(), size)
			}
			return
		}
	}

	t.Fatalf("go tool nm did not report size for runtime.gogo")
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
// adderesses that normally an OS would reserve for itself, or malformed
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
	// This currently results in a fault in the signal trampoline on
	// dragonfly/386 - see issue 7421.
	if GOOS == "dragonfly" && GOARCH == "386" {
		t.Skip("skipping test on dragonfly/386")
	}

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
	if unsafe.Sizeof(T2{}) != 8 + unsafe.Sizeof(Uintreg(0)) {
		t.Errorf("sizeof(%#v)==%d, want %d", T2{}, unsafe.Sizeof(T2{}), 8 + unsafe.Sizeof(Uintreg(0)))
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
