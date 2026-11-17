// run -goexperiment simd

//go:build linux && (amd64 || arm64)

package main

import (
	"encoding/binary"
	"fmt"
	"runtime"
	"runtime/debug"
	"syscall"
	"unsafe"
)

//go:noinline
func faultMap(dst, src []uint32, delta uint32) {
	if len(dst) > len(src) {
		return
	}
	for i := range dst {
		dst[i] = src[i] + delta
	}
}

//go:noinline
func scalarFaultAdd(a, b uint32) uint32 { return a + b }

func expectFault(f func()) {
	defer func() {
		fault := recover()
		if _, ok := fault.(runtime.Error); !ok {
			panic(fmt.Sprintf("expected a recovered memory fault, got %v", fault))
		}
	}()
	f()
}

func faultMapping(readable, n int) ([]byte, []uint32) {
	page := syscall.Getpagesize()
	mem, err := syscall.Mmap(-1, 0, 2*page, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_PRIVATE|syscall.MAP_ANON)
	if err != nil {
		panic(err)
	}
	for i := range mem[:page] {
		mem[i] = 0xa5
	}
	if err := syscall.Mprotect(mem[page:], syscall.PROT_NONE); err != nil {
		panic(err)
	}
	values := unsafe.Slice((*uint32)(unsafe.Pointer(&mem[page-4*readable])), n)
	return mem, values
}

func release(mem []byte) {
	if err := syscall.Munmap(mem); err != nil {
		panic(err)
	}
}

func sourceFault(readable int) {
	const n = 9
	mem, src := faultMapping(readable, n)
	defer release(mem)
	dst := make([]uint32, n)
	want := make([]uint32, n)
	for i := range dst {
		dst[i], want[i] = 100, 100
	}
	for i := 0; i < readable; i++ {
		src[i] = uint32(i) + 41
		want[i] = scalarFaultAdd(src[i], 7)
	}
	expectFault(func() { faultMap(dst, src, 7) })
	for i, got := range dst {
		if got != want[i] {
			panic(fmt.Sprintf("source fault after %d elements, dst[%d]=%d want %d", readable, i, got, want[i]))
		}
	}
}

func destinationFault(readable int) {
	const n = 9
	mem, dst := faultMapping(readable, n)
	defer release(mem)
	src := make([]uint32, n)
	for i := range src {
		src[i] = uint32(i) + 61
	}
	page := syscall.Getpagesize()
	want := append([]byte(nil), mem[:page]...)
	for i := 0; i < readable; i++ {
		binary.LittleEndian.PutUint32(want[page-4*readable+4*i:], scalarFaultAdd(src[i], 13))
	}
	expectFault(func() { faultMap(dst, src, 13) })
	for i, got := range mem[:page] {
		if got != want[i] {
			panic(fmt.Sprintf("destination fault after %d elements, byte[%d]=%02x want %02x", readable, i, got, want[i]))
		}
	}
}

func main() {
	original := debug.SetPanicOnFault(true)
	defer debug.SetPanicOnFault(original)
	for _, readable := range []int{0, 1, 3, 4, 5, 7, 8} {
		sourceFault(readable)
		destinationFault(readable)
	}
	debug.SetPanicOnFault(false)
	dst, src := make([]uint32, 17), make([]uint32, 17)
	for i := range src {
		src[i] = uint32(i) + 5
	}
	faultMap(dst, src, 19)
	for i, got := range dst {
		if got != scalarFaultAdd(src[i], 19) {
			panic("disabled fault recovery changed a valid map")
		}
	}
	debug.SetPanicOnFault(true)
	sourceFault(1)
	destinationFault(1)
}
