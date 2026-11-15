// asmcheck

package codegen

// wasm function carries. The amd64 lines run on the default host and pin

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
	_ = [1]byte{}[unsafe.Sizeof((*int)(nil))-8]
	_ = [1]byte{}[unsafe.Sizeof([]byte(nil))-24]
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
