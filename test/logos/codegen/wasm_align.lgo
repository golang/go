// asmcheck

package codegen

// M16-S2 (SPEC Sec 14.1, T-Sec14; #78): on GOARCH=wasm the 64-bit types
// align on 8 bytes, so 64-bit atomics stay naturally aligned with no
// forcing code. Alignof of int64, uint64, and float64 each equal 8; every
// such field after a byte field sits at an offset that is a multiple of
// 8; and every element of a [3]int64 inside a struct does too. Each
// function adds 1000 to its constant, so the expected immediate is
// unique and never collides with the 8-byte result slot offset that every
// wasm function carries. The amd64 lines run on the default host and pin
// the same constants.

import "unsafe"

type word struct {
	b byte
	i int64
	u uint64
	f float64
}

type words struct {
	b byte
	a [3]int64
}

// A wrong alignment or offset makes the constant index below invalid, so
// the file stops compiling for the target instead of passing silently.
var (
	_ = [1]byte{}[unsafe.Alignof(int64(0))-8]
	_ = [1]byte{}[unsafe.Alignof(uint64(0))-8]
	_ = [1]byte{}[unsafe.Alignof(float64(0))-8]
	_ = [1]byte{}[unsafe.Offsetof(word{}.i)%8]
	_ = [1]byte{}[unsafe.Offsetof(word{}.u)%8]
	_ = [1]byte{}[unsafe.Offsetof(word{}.f)%8]
	_ = [1]byte{}[unsafe.Offsetof(words{}.a)%8]
	_ = [1]byte{}[(unsafe.Offsetof(words{}.a)+unsafe.Sizeof(int64(0)))%8]
	_ = [1]byte{}[(unsafe.Offsetof(words{}.a)+2*unsafe.Sizeof(int64(0)))%8]
)

func alignInt64() uintptr {
	// amd64:"MOVL \\$1008, AX"
	// wasm:"I64Const \\$1008"
	return unsafe.Alignof(int64(0)) + 1000
}

func alignUint64() uintptr {
	// amd64:"MOVL \\$1008, AX"
	// wasm:"I64Const \\$1008"
	return unsafe.Alignof(uint64(0)) + 1000
}

func alignFloat64() uintptr {
	// amd64:"MOVL \\$1008, AX"
	// wasm:"I64Const \\$1008"
	return unsafe.Alignof(float64(0)) + 1000
}

func offsetInt64() uintptr {
	// amd64:"MOVL \\$1008, AX"
	// wasm:"I64Const \\$1008"
	return unsafe.Offsetof(word{}.i) + 1000
}

func offsetUint64() uintptr {
	// amd64:"MOVL \\$1016, AX"
	// wasm:"I64Const \\$1016"
	return unsafe.Offsetof(word{}.u) + 1000
}

func offsetFloat64() uintptr {
	// amd64:"MOVL \\$1024, AX"
	// wasm:"I64Const \\$1024"
	return unsafe.Offsetof(word{}.f) + 1000
}

func offsetElem0() uintptr {
	// amd64:"MOVL \\$1008, AX"
	// wasm:"I64Const \\$1008"
	return unsafe.Offsetof(words{}.a) + 1000
}

func offsetElem1() uintptr {
	// amd64:"MOVL \\$1016, AX"
	// wasm:"I64Const \\$1016"
	return unsafe.Offsetof(words{}.a) + 1*unsafe.Sizeof(int64(0)) + 1000
}

func offsetElem2() uintptr {
	// amd64:"MOVL \\$1024, AX"
	// wasm:"I64Const \\$1024"
	return unsafe.Offsetof(words{}.a) + 2*unsafe.Sizeof(int64(0)) + 1000
}

func main() {}
