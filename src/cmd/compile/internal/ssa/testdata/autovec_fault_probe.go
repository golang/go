//go:build linux && (amd64 || arm64)

package main

import (
	"fmt"
	"runtime/debug"
	"syscall"
	"unsafe"
)

//go:noinline
func mappedCopy(dst, src []uint32) {
	if len(dst) > len(src) {
		return
	}
	for i := range dst {
		dst[i] = src[i] + 1
	}
}

//go:noinline
func adjacentStores(dst *[2]uint32) {
	dst[0] = 11
	dst[1] = 22
}

func caught(f func()) (panicked bool) {
	defer func() { panicked = recover() != nil }()
	f()
	return false
}

func main() {
	page := syscall.Getpagesize()
	mem, err := syscall.Mmap(-1, 0, 2*page, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_PRIVATE|syscall.MAP_ANON)
	if err != nil {
		panic(err)
	}
	defer func() {
		if err := syscall.Munmap(mem); err != nil {
			panic(err)
		}
	}()
	first := (*uint32)(unsafe.Pointer(&mem[page-4]))
	*first = 41
	src := unsafe.Slice(first, 4)
	if err := syscall.Mprotect(mem[page:], syscall.PROT_NONE); err != nil {
		panic(err)
	}
	old := debug.SetPanicOnFault(true)
	defer debug.SetPanicOnFault(old)

	dst := []uint32{100, 100, 100, 100}
	panicked := caught(func() { mappedCopy(dst, src) })
	fmt.Printf("map recovered=%t dst=%v\n", panicked, dst)

	*first = 41
	panicked = caught(func() { adjacentStores((*[2]uint32)(unsafe.Pointer(first))) })
	fmt.Printf("adjacent recovered=%t first=%d\n", panicked, *first)
}
