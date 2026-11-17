// run -goexperiment simd

//go:build amd64

package main

import (
	"fmt"
	"os"
	"simd/archsimd"
)

var calls int

//go:noinline
func recordCall() { calls++ }

//go:noinline
func guarded128(s []uint8) uint8 {
	var result uint8
	if archsimd.X86.AVX2() {
		var value archsimd.Uint8x16
		if len(s) >= 16 {
			value = value.Add(archsimd.LoadUint8x16Array((*[16]uint8)(s)))
		}
		var lanes [16]uint8
		value.StoreArray(&lanes)
		result += lanes[0]
	} else if len(s) >= 16 {
		result += s[0]
	}
	recordCall()
	if archsimd.X86.AVX2() {
		var value archsimd.Uint8x16
		if len(s) >= 16 {
			value = value.Add(archsimd.LoadUint8x16Array((*[16]uint8)(s)))
		}
		var lanes [16]uint8
		value.StoreArray(&lanes)
		result += lanes[0]
	} else if len(s) >= 16 {
		result += s[0]
	}
	return result
}

//go:noinline
func guarded256(s []uint8) uint8 {
	var result uint8
	if archsimd.X86.AVX2() {
		var value archsimd.Uint8x32
		if len(s) >= 32 {
			value = value.Add(archsimd.LoadUint8x32Array((*[32]uint8)(s)))
		}
		var lanes [32]uint8
		value.StoreArray(&lanes)
		result += lanes[0]
	} else if len(s) >= 32 {
		result += s[0]
	}
	recordCall()
	if archsimd.X86.AVX2() {
		var value archsimd.Uint8x32
		if len(s) >= 32 {
			value = value.Add(archsimd.LoadUint8x32Array((*[32]uint8)(s)))
		}
		var lanes [32]uint8
		value.StoreArray(&lanes)
		result += lanes[0]
	} else if len(s) >= 32 {
		result += s[0]
	}
	return result
}

//go:noinline
func guarded512(s []uint8) uint8 {
	var result uint8
	if archsimd.X86.AVX512() {
		var value archsimd.Uint8x64
		if len(s) >= 64 {
			value = value.Add(archsimd.LoadUint8x64Array((*[64]uint8)(s)))
		}
		var lanes [64]uint8
		value.StoreArray(&lanes)
		result += lanes[0]
	} else if len(s) >= 64 {
		result += s[0]
	}
	recordCall()
	if archsimd.X86.AVX512() {
		var value archsimd.Uint8x64
		if len(s) >= 64 {
			value = value.Add(archsimd.LoadUint8x64Array((*[64]uint8)(s)))
		}
		var lanes [64]uint8
		value.StoreArray(&lanes)
		result += lanes[0]
	} else if len(s) >= 64 {
		result += s[0]
	}
	return result
}

func check(width int, fn func([]uint8) uint8) {
	data := make([]uint8, width+1)
	for i := range data {
		data[i] = uint8(i*17 + 3)
	}
	before := calls
	for _, n := range []int{-1, 0, 1, width - 1, width, width + 1} {
		var s []uint8
		if n >= 0 {
			s = data[:n]
		}
		var want uint8
		if len(s) >= width {
			want = s[0] + s[0]
		}
		if got := fn(s); got != want {
			panic(fmt.Sprintf("width=%d len=%d got=%d want=%d", width, n, got, want))
		}
	}
	if calls-before != 6 {
		panic(fmt.Sprintf("width=%d calls=%d want=6", width, calls-before))
	}
}

func main() {
	choice := "all"
	if len(os.Args) == 2 {
		choice = os.Args[1]
	}
	switch choice {
	case "all", "128", "256", "512":
	default:
		panic("invalid width")
	}
	if choice == "all" || choice == "128" {
		check(16, guarded128)
	}
	if choice == "all" || choice == "256" {
		check(32, guarded256)
	}
	if choice == "all" || choice == "512" {
		check(64, guarded512)
	}
	fmt.Println("SIMD guards hold")
}
