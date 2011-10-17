package runtime_test

import (
	"runtime"
	"testing"
	"unsafe"
)

func TestGcSys(t *testing.T) {
	for i := 0; i < 1000000; i++ {
		workthegc()
	}

	// Should only be using a few MB.
	runtime.UpdateMemStats()
	sys := runtime.MemStats.Sys
	t.Logf("using %d MB", sys>>20)
	if sys > 10e6 {
		t.Fatalf("using too much memory: %d MB", sys>>20)
	}
}

func workthegc() []byte {
	return make([]byte, 1029)
}

func TestGcUintptr(t *testing.T) {
	p1 := unsafe.Pointer(new(int))
	*(*int)(unsafe.Pointer(p1)) = 42
	p2 := uintptr(unsafe.Pointer(new(int)))
	*(*int)(unsafe.Pointer(p2)) = 42
	var a1 [1]unsafe.Pointer
	a1[0] = unsafe.Pointer(new(int))
	*(*int)(unsafe.Pointer(a1[0])) = 42
	var a2 [1]uintptr
	a2[0] = uintptr(unsafe.Pointer(new(int)))
	*(*int)(unsafe.Pointer(a2[0])) = 42
	s1 := make([]unsafe.Pointer, 1)
	s1[0] = unsafe.Pointer(new(int))
	*(*int)(unsafe.Pointer(s1[0])) = 42
	s2 := make([]uintptr, 1)
	s2[0] = uintptr(unsafe.Pointer(new(int)))
	*(*int)(unsafe.Pointer(s2[0])) = 42
	m1 := make(map[int]unsafe.Pointer)
	m1[0] = unsafe.Pointer(new(int))
	*(*int)(unsafe.Pointer(m1[0])) = 42
	m2 := make(map[int]uintptr)
	m2[0] = uintptr(unsafe.Pointer(new(int)))
	*(*int)(unsafe.Pointer(m2[0])) = 42
	c1 := make(chan unsafe.Pointer, 1)
	func() {
		p := new(int)
		*p = 42
		c1 <- unsafe.Pointer(p)
	}()
	c2 := make(chan uintptr, 1)
	func() {
		p := new(int)
		*p = 42
		c2 <- uintptr(unsafe.Pointer(p))
	}()

	runtime.GC()

	if p, _ := runtime.Lookup((*byte)(unsafe.Pointer(p1))); p == nil || *(*int)(unsafe.Pointer(p)) != 42 {
		t.Fatalf("p1 is freed")
	}
	if p, _ := runtime.Lookup((*byte)(unsafe.Pointer(p2))); p == nil || *(*int)(unsafe.Pointer(p)) != 42 {
		t.Fatalf("p2 is freed")
	}
	if p, _ := runtime.Lookup((*byte)(unsafe.Pointer(a1[0]))); p == nil || *(*int)(unsafe.Pointer(p)) != 42 {
		t.Fatalf("a1[0] is freed")
	}
	if p, _ := runtime.Lookup((*byte)(unsafe.Pointer(a2[0]))); p == nil || *(*int)(unsafe.Pointer(p)) != 42 {
		t.Fatalf("a2[0] is freed")
	}
	if p, _ := runtime.Lookup((*byte)(unsafe.Pointer(s1[0]))); p == nil || *(*int)(unsafe.Pointer(p)) != 42 {
		t.Fatalf("s1[0] is freed")
	}
	if p, _ := runtime.Lookup((*byte)(unsafe.Pointer(s2[0]))); p == nil || *(*int)(unsafe.Pointer(p)) != 42 {
		t.Fatalf("s2[0] is freed")
	}
	if p, _ := runtime.Lookup((*byte)(unsafe.Pointer(m1[0]))); p == nil || *(*int)(unsafe.Pointer(p)) != 42 {
		t.Fatalf("m1[0] is freed")
	}
	if p, _ := runtime.Lookup((*byte)(unsafe.Pointer(m2[0]))); p == nil || *(*int)(unsafe.Pointer(p)) != 42 {
		t.Fatalf("m2[0] is freed")
	}
	if p, _ := runtime.Lookup((*byte)(unsafe.Pointer(<-c1))); p == nil || *(*int)(unsafe.Pointer(p)) != 42 {
		t.Fatalf("<-c1 is freed")
	}
	if p, _ := runtime.Lookup((*byte)(unsafe.Pointer(<-c2))); p == nil || *(*int)(unsafe.Pointer(p)) != 42 {
		t.Fatalf("<-c2 is freed")
	}
}
