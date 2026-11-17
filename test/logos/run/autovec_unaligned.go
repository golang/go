// run -goexperiment simd

//go:build amd64 || arm64

package main

import (
	"encoding/binary"
	"fmt"
	"unsafe"
)

//go:noinline
func unalignedAdd(dst, src []uint32) {
	if len(dst) > len(src) {
		return
	}
	for i := range dst {
		dst[i] += src[i]
	}
}

//go:noinline
func scalarUnalignedAdd(backing []byte, dst, src, n int) {
	for i := 0; i < n; i++ {
		d, s := dst+4*i, src+4*i
		value := binary.LittleEndian.Uint32(backing[d:]) + binary.LittleEndian.Uint32(backing[s:])
		binary.LittleEndian.PutUint32(backing[d:], value)
	}
}

func main() {
	for _, n := range []int{0, 1, 3, 4, 5, 8, 9, 17} {
		for _, distance := range []int{0, 1, 2, 3, 4*n - 1, 4 * n, 4*n + 1, 4*n + 3} {
			if distance < 0 {
				continue
			}
			for _, reverse := range []bool{false, true} {
				// The first address is unaligned. The other moves through
				// aligned and unaligned positions in both address orders.
				first, second := 9, 9+distance
				if reverse {
					first, second = second, first
				}
				backing := make([]byte, 32+distance+4*n)
				for i := range backing {
					backing[i] = byte(uint64(i) * 0x9e3779b97f4a7c15 >> 24)
				}
				want := append([]byte(nil), backing...)
				scalarUnalignedAdd(want, first, second, n)
				dst := unsafe.Slice((*uint32)(unsafe.Pointer(&backing[first])), n)
				src := unsafe.Slice((*uint32)(unsafe.Pointer(&backing[second])), n)
				unalignedAdd(dst, src)
				for i, got := range backing {
					if got != want[i] {
						panic(fmt.Sprintf("n=%d distance=%d reverse=%t byte=%d: got %02x want %02x", n, distance, reverse, i, got, want[i]))
					}
				}
			}
		}
	}
}
