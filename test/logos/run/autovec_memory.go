// run -goexperiment simd

//go:build linux && (amd64 || arm64)

package main

import (
	"fmt"
	"syscall"
	"unsafe"
)

type integer interface {
	~int8 | ~uint8 | ~int16 | ~uint16 | ~int32 | ~uint32 | ~int64 | ~uint64
}

//go:noinline
func memoryMap[T integer](dst, a, b []T, k T) {
	if len(dst) > len(a) || len(dst) > len(b) {
		return
	}
	for i := range dst {
		x, y := a[i], b[i]
		s := x + k
		dst[i] = (s ^ (y - k)) + (s & (x | y))
	}
}

//go:noinline
func scalarMemoryStep[T integer](x, y, k T) T {
	s := x + k
	return (s ^ (y - k)) + (s & (x | y))
}

func reference[T integer](dst, a, b []T, k T) {
	for i := range dst {
		dst[i] = scalarMemoryStep(a[i], b[i], k)
	}
}

func nextRandom(state *uint64) uint64 {
	x := *state
	x ^= x << 13
	x ^= x >> 7
	x ^= x << 17
	*state = x
	return x
}

func fillBytes(dst []byte, state uint64) {
	for i := range dst {
		dst[i] = byte(nextRandom(&state))
	}
}

func byteView[T integer](backing []byte, offset, n int) []T {
	if n == 0 {
		return nil
	}
	return unsafe.Slice((*T)(unsafe.Pointer(&backing[offset])), n)
}

func checkBytes(name, layout string, n, iteration int, got, want []byte) {
	for i, v := range got {
		if v != want[i] {
			panic(fmt.Sprintf("%s %s n=%d case=%d byte=%d: got %02x want %02x", name, layout, n, iteration, i, v, want[i]))
		}
	}
}

type guarded[T integer] struct {
	mapping []byte
	data    []byte
	values  []T
	offset  int
}

func newGuarded[T integer](n int, atEnd bool, seed uint64) guarded[T] {
	page := syscall.Getpagesize()
	mem, err := syscall.Mmap(-1, 0, 3*page, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_PRIVATE|syscall.MAP_ANON)
	if err != nil {
		panic(fmt.Sprintf("mmap: %v", err))
	}
	if err := syscall.Mprotect(mem[:page], syscall.PROT_NONE); err != nil {
		panic(fmt.Sprintf("protect head: %v", err))
	}
	if err := syscall.Mprotect(mem[2*page:], syscall.PROT_NONE); err != nil {
		panic(fmt.Sprintf("protect tail: %v", err))
	}
	var zero T
	offset := 0
	if atEnd {
		offset = page - n*int(unsafe.Sizeof(zero))
	}
	data := mem[page : 2*page]
	fillBytes(data, seed)
	// For an empty tail slice this pointer is inside the protected page.
	values := unsafe.Slice((*T)(unsafe.Pointer(&mem[page+offset])), n)
	return guarded[T]{mapping: mem, data: data, values: values, offset: offset}
}

func (g guarded[T]) close() {
	if err := syscall.Munmap(g.mapping); err != nil {
		panic(fmt.Sprintf("munmap: %v", err))
	}
}

func checkPages[T integer](name string) {
	var zero T
	lanes := 16 / int(unsafe.Sizeof(zero))
	lengths := []int{0, 1, lanes - 1, lanes, lanes + 1, 2*lanes - 1, 2 * lanes, 2*lanes + 1, 5*lanes + 3}
	seed := uint64(0xfedcba9876543281)
	k := T(seed)
	for _, n := range lengths {
		for layout := 0; layout < 8; layout++ {
			dst := newGuarded[T](n, layout&1 == 0, seed+uint64(n+layout))
			a := newGuarded[T](n, layout&2 == 0, seed^0x9e3779b97f4a7c15)
			b := newGuarded[T](n, layout&4 == 0, seed^0xd1b54a32d192ed03)
			wantDst := append([]byte(nil), dst.data...)
			wantA := append([]byte(nil), a.data...)
			wantB := append([]byte(nil), b.data...)
			reference(byteView[T](wantDst, dst.offset, n), a.values, b.values, k)
			memoryMap(dst.values, a.values, b.values, k)
			checkBytes(name, "page destination", n, layout, dst.data, wantDst)
			checkBytes(name, "page source A", n, layout, a.data, wantA)
			checkBytes(name, "page source B", n, layout, b.data, wantB)
			dst.close()
			a.close()
			b.close()
		}
	}
}

func checkAliases[T integer](name string) {
	var zero T
	size := int(unsafe.Sizeof(zero))
	lanes := 16 / size
	state := uint64(0x9e3779b97f4a7c15)
	keys := []uint64{0, 1, ^uint64(0), 0x8000000000000000, 0x8000800080008080, 0x7fff7fff7fff7f7f}
	for iteration := 0; iteration < 384; iteration++ {
		n := int(nextRandom(&state) % uint64(5*lanes+4))
		span := n * size
		backing := make([]byte, 3*span+64)
		fillBytes(backing, nextRandom(&state))
		offsets := len(backing) - span + 1
		dst := int(nextRandom(&state) % uint64(offsets))
		a := int(nextRandom(&state) % uint64(offsets))
		b := int(nextRandom(&state) % uint64(offsets))
		switch layout := iteration % 16; layout {
		case 0:
			a, b = dst, dst
		case 1:
			a = dst
		case 2:
			b = dst
		case 3:
			b = a
		case 4:
			dst, a, b = 17, 18, 2*span+33
		case 5:
			dst, a, b = 18, 17, 2*span+33
		case 6, 7, 8, 9, 10, 11:
			distance := n / lanes * 16
			distance += (layout-6)/2 - 1
			if distance < 0 {
				distance = 0
			}
			dst, a, b = 17, 17+distance, 2*span+33
			if layout%2 != 0 {
				dst, a = a, dst
			}
		}
		key := nextRandom(&state)
		if iteration%2 == 0 {
			key = keys[(iteration/2)%len(keys)]
		}
		k := T(key)
		want := append([]byte(nil), backing...)
		reference(byteView[T](want, dst, n), byteView[T](want, a, n), byteView[T](want, b, n), k)
		memoryMap(byteView[T](backing, dst, n), byteView[T](backing, a, n), byteView[T](backing, b, n), k)
		checkBytes(name, "aliases", n, iteration, backing, want)
	}
}

func checkType[T integer](name string) {
	checkPages[T](name)
	checkAliases[T](name)
}

func main() {
	checkType[int8]("int8")
	checkType[uint8]("uint8")
	checkType[int16]("int16")
	checkType[uint16]("uint16")
	checkType[int32]("int32")
	checkType[uint32]("uint32")
	checkType[int64]("int64")
	checkType[uint64]("uint64")
}
