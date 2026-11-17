// asmcheck -goexperiment simd

package main

//go:noinline
func negate8(x []int8) {
	for i := range x {
		// amd64/v1:-"VPSUBB"
		// amd64/v2:-"VPSUBB"
		// amd64/v3:"VPSUBB"
		// amd64/v4:"VPSUBB"
		// arm64:"V(SUB|NEG) V[0-9]+\\.B16"
		x[i] = -x[i]
	}
}

//go:noinline
func complement8(x []uint8) {
	for i := range x {
		// amd64/v3:"VPXOR"
		// amd64/v4:"VPXOR"
		// arm64:"V(EOR|NOT) V[0-9]+\\.B16"
		x[i] = ^x[i]
	}
}

//go:noinline
func negate16(x []int16) {
	for i := range x {
		// amd64/v3:"VPSUBW"
		// amd64/v4:"VPSUBW"
		// arm64:"V(SUB|NEG) V[0-9]+\\.H8"
		x[i] = -x[i]
	}
}

//go:noinline
func complement16(x []uint16) {
	for i := range x {
		// amd64/v3:"VPXOR"
		// amd64/v4:"VPXOR"
		// arm64:"V(EOR|NOT) V[0-9]+\\.B16"
		x[i] = ^x[i]
	}
}

//go:noinline
func negate32(x []int32) {
	for i := range x {
		// amd64/v3:"VPSUBD"
		// amd64/v4:"VPSUBD"
		// arm64:"V(SUB|NEG) V[0-9]+\\.S4"
		x[i] = -x[i]
	}
}

//go:noinline
func complement32(x []uint32) {
	for i := range x {
		// amd64/v3:"VPXOR"
		// amd64/v4:"VPXOR"
		// arm64:"V(EOR|NOT) V[0-9]+\\.B16"
		x[i] = ^x[i]
	}
}

//go:noinline
func negate64(x []int64) {
	for i := range x {
		// amd64/v3:"VPSUBQ"
		// amd64/v4:"VPSUBQ"
		// arm64:"V(SUB|NEG) V[0-9]+\\.D2"
		x[i] = -x[i]
	}
}

//go:noinline
func complement64(x []uint64) {
	for i := range x {
		// amd64/v3:"VPXOR"
		// amd64/v4:"VPXOR"
		// arm64:"V(EOR|NOT) V[0-9]+\\.B16"
		x[i] = ^x[i]
	}
}

//go:noinline
func doubleBytes(x []uint8) {
	for i := range x {
		// amd64/v3:"VPADDB"
		// amd64/v4:"VPADDB"
		// arm64:"VADD V[0-9]+\\.B16"
		x[i] = x[i] + x[i]
	}
}

//go:noinline
func leftBytesOne(x []uint8) {
	for i := range x {
		// amd64/v3:"VPADDB"
		// amd64/v4:"VPADDB"
		// arm64:"VADD V[0-9]+\\.B16"
		x[i] = x[i] << 1
	}
}

//go:noinline
func left16(x []uint16) {
	for i := range x {
		// amd64/v3:"VPSLLW"
		// amd64/v4:"VPSLLW"
		// arm64:"V(SSHL|USHL|SHL) V[0-9]+\\.H8"
		x[i] = x[i] << 3
	}
}

//go:noinline
func unsignedRight16(x []uint16) {
	for i := range x {
		// amd64/v3:"VPSRLW"
		// amd64/v4:"VPSRLW"
		// arm64:"V(USHL|USHR) V[0-9]+\\.H8"
		x[i] = x[i] >> 3
	}
}

//go:noinline
func signedRight16(x []int16) {
	for i := range x {
		// amd64/v3:"VPSRAW"
		// amd64/v4:"VPSRAW"
		// arm64:"V(SSHL|SSHR) V[0-9]+\\.H8"
		x[i] = x[i] >> 3
	}
}

//go:noinline
func left32(x []uint32) {
	for i := range x {
		// amd64/v3:"VPSLLD"
		// amd64/v4:"VPSLLD"
		// arm64:"V(SSHL|USHL|SHL) V[0-9]+\\.S4"
		x[i] = x[i] << 3
	}
}

//go:noinline
func unsignedRight32(x []uint32) {
	for i := range x {
		// amd64/v3:"VPSRLD"
		// amd64/v4:"VPSRLD"
		// arm64:"V(USHL|USHR) V[0-9]+\\.S4"
		x[i] = x[i] >> 3
	}
}

//go:noinline
func signedRight32(x []int32) {
	for i := range x {
		// amd64/v3:"VPSRAD"
		// amd64/v4:"VPSRAD"
		// arm64:"V(SSHL|SSHR) V[0-9]+\\.S4"
		x[i] = x[i] >> 3
	}
}

//go:noinline
func left64(x []uint64) {
	for i := range x {
		// amd64/v3:"VPSLLQ"
		// amd64/v4:"VPSLLQ"
		// arm64:"V(SSHL|USHL|SHL) V[0-9]+\\.D2"
		x[i] = x[i] << 3
	}
}

//go:noinline
func unsignedRight64(x []uint64) {
	for i := range x {
		// amd64/v3:"VPSRLQ"
		// amd64/v4:"VPSRLQ"
		// arm64:"V(USHL|USHR) V[0-9]+\\.D2"
		x[i] = x[i] >> 3
	}
}

//go:noinline
func generalByteShift(x []uint8) {
	for i := range x {
		// amd64/v3:-"VPSLL"
		// amd64/v4:-"VPSLL"
		// arm64:-"V(SSHL|USHL|SHL)"
		x[i] = x[i] << 3
	}
}

//go:noinline
func signedRight64(x []int64) {
	for i := range x {
		// amd64/v3:-"VPSRAQ"
		// amd64/v4:-"VPSRAQ"
		// arm64:-"V(SSHL|SSHR)"
		x[i] = x[i] >> 3
	}
}

//go:noinline
func variableShift(x []uint32, count uint64) {
	for i := range x {
		// amd64/v3:-"VPSLLD"
		// amd64/v4:-"VPSLLD"
		// arm64:-"V(SSHL|USHL|SHL)"
		x[i] = x[i] << count
	}
}

func main() {}
