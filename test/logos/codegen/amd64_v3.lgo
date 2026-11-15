// asmcheck

package codegen

import "math/bits"

func rolImm64(x uint64) uint64 {
	// amd64/v1:"ROLQ [$]7," -"RORX"
	// amd64/v2:"ROLQ [$]7," -"RORX"
	// amd64/v3:"RORXQ [$]57," -"ROLQ"
	// amd64/v4:"RORXQ [$]57," -"ROLQ"
	return x<<7 | x>>57
}

func rolImm32(x uint32) uint32 {
	// amd64/v1:"ROLL [$]7," -"RORX"
	// amd64/v2:"ROLL [$]7," -"RORX"
	// amd64/v3:"RORXL [$]25," -"ROLL"
	// amd64/v4:"RORXL [$]25," -"ROLL"
	return x<<7 | x>>25
}

func rolImm64Low(x uint64) uint64 {
	// amd64/v1:"ROLQ [$]1," -"RORX"
	// amd64/v2:"ROLQ [$]1," -"RORX"
	// amd64/v3:"RORXQ [$]63," -"ROLQ"
	// amd64/v4:"RORXQ [$]63," -"ROLQ"
	return bits.RotateLeft64(x, 1)
}

func rolImm64High(x uint64) uint64 {
	// amd64/v1:"ROLQ [$]63," -"RORX"
	// amd64/v2:"ROLQ [$]63," -"RORX"
	// amd64/v3:"RORXQ [$]1," -"ROLQ"
	// amd64/v4:"RORXQ [$]1," -"ROLQ"
	return bits.RotateLeft64(x, 63)
}

func rolImm64Wrap(x uint64) uint64 {
	// amd64/v1:"ROLQ [$]1," -"RORX"
	// amd64/v2:"ROLQ [$]1," -"RORX"
	// amd64/v3:"RORXQ [$]63," -"ROLQ"
	// amd64/v4:"RORXQ [$]63," -"ROLQ"
	return bits.RotateLeft64(x, 65)
}

func rolImm64Negative(x uint64) uint64 {
	// amd64/v1:"ROLQ [$]63," -"RORX"
	// amd64/v2:"ROLQ [$]63," -"RORX"
	// amd64/v3:"RORXQ [$]1," -"ROLQ"
	// amd64/v4:"RORXQ [$]1," -"ROLQ"
	return bits.RotateLeft64(x, -1)
}

func rolImm64Large(x uint64) uint64 {
	// amd64/v1:"ROLQ [$]3," -"RORX"
	// amd64/v2:"ROLQ [$]3," -"RORX"
	// amd64/v3:"RORXQ [$]61," -"ROLQ"
	// amd64/v4:"RORXQ [$]61," -"ROLQ"
	return bits.RotateLeft64(x, 1<<40+3)
}

func rolImm32Low(x uint32) uint32 {
	// amd64/v1:"ROLL [$]1," -"RORX"
	// amd64/v2:"ROLL [$]1," -"RORX"
	// amd64/v3:"RORXL [$]31," -"ROLL"
	// amd64/v4:"RORXL [$]31," -"ROLL"
	return bits.RotateLeft32(x, 1)
}

func rolImm32High(x uint32) uint32 {
	// amd64/v1:"ROLL [$]31," -"RORX"
	// amd64/v2:"ROLL [$]31," -"RORX"
	// amd64/v3:"RORXL [$]1," -"ROLL"
	// amd64/v4:"RORXL [$]1," -"ROLL"
	return bits.RotateLeft32(x, 31)
}

func rolImm32Wrap(x uint32) uint32 {
	// amd64/v1:"ROLL [$]1," -"RORX"
	// amd64/v2:"ROLL [$]1," -"RORX"
	// amd64/v3:"RORXL [$]31," -"ROLL"
	// amd64/v4:"RORXL [$]31," -"ROLL"
	return bits.RotateLeft32(x, 33)
}

func rolImm32Negative(x uint32) uint32 {
	// amd64/v1:"ROLL [$]31," -"RORX"
	// amd64/v2:"ROLL [$]31," -"RORX"
	// amd64/v3:"RORXL [$]1," -"ROLL"
	// amd64/v4:"RORXL [$]1," -"ROLL"
	return bits.RotateLeft32(x, -1)
}

func rolZero64(x uint64) uint64 {
	// amd64:-"ROLQ" -"RORQ" -"RORX"
	return bits.RotateLeft64(x, 0)
}

func rolFull64(x uint64) uint64 {
	// amd64:-"ROLQ" -"RORQ" -"RORX"
	return bits.RotateLeft64(x, 64)
}

func rolFullNegative64(x uint64) uint64 {
	// amd64:-"ROLQ" -"RORQ" -"RORX"
	return bits.RotateLeft64(x, -64)
}

func rolZero32(x uint32) uint32 {
	// amd64:-"ROLL" -"RORL" -"RORX"
	return bits.RotateLeft32(x, 0)
}

func rolFull32(x uint32) uint32 {
	// amd64:-"ROLL" -"RORL" -"RORX"
	return bits.RotateLeft32(x, 32)
}

func rolImm16(x uint16) uint16 {
	// amd64:"ROLW [$]3," -"RORX"
	return x<<3 | x>>13
}

func rolImm8(x uint8) uint8 {
	// amd64:"ROLB [$]3," -"RORX"
	return x<<3 | x>>5
}

func rolVar64(x uint64, n int) uint64 {
	// amd64/v3:"ROLQ" -"RORX"
	// amd64/v4:"ROLQ" -"RORX"
	return bits.RotateLeft64(x, n)
}

func rolVar32(x uint32, n int) uint32 {
	// amd64/v3:"ROLL" -"RORX"
	// amd64/v4:"ROLL" -"RORX"
	return bits.RotateLeft32(x, n)
}

func rorVar64(x uint64, z uint) uint64 {
	z &= 63
	// amd64/v3:"RORQ" -"AND" -"RORX"
	// amd64/v4:"RORQ" -"AND" -"RORX"
	return x>>z | x<<(64-z)
}

func rorVar32(x uint32, z uint) uint32 {
	z &= 31
	// amd64/v3:"RORL" -"AND" -"RORX"
	// amd64/v4:"RORL" -"AND" -"RORX"
	return x>>z | x<<(32-z)
}

func shlVar64(x uint64, n uint) uint64 {
	// amd64/v1:"SHLQ" -"SHLX"
	// amd64/v2:"SHLQ" -"SHLX"
	// amd64/v3:"SHLXQ" "CMPQ" "SBBQ" -"SHLQ"
	// amd64/v4:"SHLXQ" "CMPQ" "SBBQ" -"SHLQ"
	return x << n
}

func shrVar64(x uint64, n uint) uint64 {
	// amd64/v1:"SHRQ" -"SHRX"
	// amd64/v2:"SHRQ" -"SHRX"
	// amd64/v3:"SHRXQ" "CMPQ" "SBBQ" -"SHRQ"
	// amd64/v4:"SHRXQ" "CMPQ" "SBBQ" -"SHRQ"
	return x >> n
}

func sarVar64(x int64, n uint) int64 {
	// amd64/v1:"SARQ" -"SARX"
	// amd64/v2:"SARQ" -"SARX"
	// amd64/v3:"SARXQ" "CMPQ" "SBBQ" -"SARQ"
	// amd64/v4:"SARXQ" "CMPQ" "SBBQ" -"SARQ"
	return x >> n
}

func shlVar32(x uint32, n uint) uint32 {
	// amd64/v1:"SHLL" -"SHLX"
	// amd64/v2:"SHLL" -"SHLX"
	// amd64/v3:"SHLXL" "CMPQ" "SBBL" -"SHLL"
	// amd64/v4:"SHLXL" "CMPQ" "SBBL" -"SHLL"
	return x << n
}

func shrVar32(x uint32, n uint) uint32 {
	// amd64/v1:"SHRL" -"SHRX"
	// amd64/v2:"SHRL" -"SHRX"
	// amd64/v3:"SHRXL" "CMPQ" "SBBL" -"SHRL"
	// amd64/v4:"SHRXL" "CMPQ" "SBBL" -"SHRL"
	return x >> n
}

func sarVar32(x int32, n uint) int32 {
	// amd64/v1:"SARL" -"SARX"
	// amd64/v2:"SARL" -"SARX"
	// amd64/v3:"SARXL" "CMPQ" "SBBQ" -"SARL"
	// amd64/v4:"SARXL" "CMPQ" "SBBQ" -"SARL"
	return x >> n
}

func cmovSigned(c int) int {
	x := c + 4
	if x < 0 {
		x = 182
	}
	// amd64/v1:"CMOVQLT"
	// amd64/v2:"CMOVQLT"
	// amd64/v3:"CMOVQLT"
	// amd64/v4:"CMOVQLT"
	return x
}

func cmovUnsigned32(x, y uint32) uint32 {
	if x < y {
		x = -y
	}
	// amd64/v1:"CMOVL(HI|CS)"
	// amd64/v2:"CMOVL(HI|CS)"
	// amd64/v3:"CMOVL(HI|CS)"
	// amd64/v4:"CMOVL(HI|CS)"
	return x
}

func andNot64(x, y uint64) uint64 {
	// amd64/v1:-"ANDNQ"
	// amd64/v2:-"ANDNQ"
	// amd64/v3:"ANDNQ"
	// amd64/v4:"ANDNQ"
	return x &^ y
}

func clearLowest64(x uint64) uint64 {
	// amd64/v1:-"BLSRQ"
	// amd64/v2:-"BLSRQ"
	// amd64/v3:"BLSRQ"
	// amd64/v4:"BLSRQ"
	return x & (x - 1)
}

func mulHigh64(x, y uint64) uint64 {
	// amd64/v1:-"MULXQ"
	// amd64/v2:-"MULXQ"
	hi, _ := bits.Mul64(x, y)
	return hi
}
