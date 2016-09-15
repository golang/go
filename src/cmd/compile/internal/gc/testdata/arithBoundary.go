package main

import "fmt"

type utd64 struct {
	a, b                    uint64
	add, sub, mul, div, mod uint64
}
type itd64 struct {
	a, b                    int64
	add, sub, mul, div, mod int64
}
type utd32 struct {
	a, b                    uint32
	add, sub, mul, div, mod uint32
}
type itd32 struct {
	a, b                    int32
	add, sub, mul, div, mod int32
}
type utd16 struct {
	a, b                    uint16
	add, sub, mul, div, mod uint16
}
type itd16 struct {
	a, b                    int16
	add, sub, mul, div, mod int16
}
type utd8 struct {
	a, b                    uint8
	add, sub, mul, div, mod uint8
}
type itd8 struct {
	a, b                    int8
	add, sub, mul, div, mod int8
}

//go:noinline
func add_uint64_ssa(a, b uint64) uint64 {
	return a + b
}

//go:noinline
func sub_uint64_ssa(a, b uint64) uint64 {
	return a - b
}

//go:noinline
func div_uint64_ssa(a, b uint64) uint64 {
	return a / b
}

//go:noinline
func mod_uint64_ssa(a, b uint64) uint64 {
	return a % b
}

//go:noinline
func mul_uint64_ssa(a, b uint64) uint64 {
	return a * b
}

//go:noinline
func add_int64_ssa(a, b int64) int64 {
	return a + b
}

//go:noinline
func sub_int64_ssa(a, b int64) int64 {
	return a - b
}

//go:noinline
func div_int64_ssa(a, b int64) int64 {
	return a / b
}

//go:noinline
func mod_int64_ssa(a, b int64) int64 {
	return a % b
}

//go:noinline
func mul_int64_ssa(a, b int64) int64 {
	return a * b
}

//go:noinline
func add_uint32_ssa(a, b uint32) uint32 {
	return a + b
}

//go:noinline
func sub_uint32_ssa(a, b uint32) uint32 {
	return a - b
}

//go:noinline
func div_uint32_ssa(a, b uint32) uint32 {
	return a / b
}

//go:noinline
func mod_uint32_ssa(a, b uint32) uint32 {
	return a % b
}

//go:noinline
func mul_uint32_ssa(a, b uint32) uint32 {
	return a * b
}

//go:noinline
func add_int32_ssa(a, b int32) int32 {
	return a + b
}

//go:noinline
func sub_int32_ssa(a, b int32) int32 {
	return a - b
}

//go:noinline
func div_int32_ssa(a, b int32) int32 {
	return a / b
}

//go:noinline
func mod_int32_ssa(a, b int32) int32 {
	return a % b
}

//go:noinline
func mul_int32_ssa(a, b int32) int32 {
	return a * b
}

//go:noinline
func add_uint16_ssa(a, b uint16) uint16 {
	return a + b
}

//go:noinline
func sub_uint16_ssa(a, b uint16) uint16 {
	return a - b
}

//go:noinline
func div_uint16_ssa(a, b uint16) uint16 {
	return a / b
}

//go:noinline
func mod_uint16_ssa(a, b uint16) uint16 {
	return a % b
}

//go:noinline
func mul_uint16_ssa(a, b uint16) uint16 {
	return a * b
}

//go:noinline
func add_int16_ssa(a, b int16) int16 {
	return a + b
}

//go:noinline
func sub_int16_ssa(a, b int16) int16 {
	return a - b
}

//go:noinline
func div_int16_ssa(a, b int16) int16 {
	return a / b
}

//go:noinline
func mod_int16_ssa(a, b int16) int16 {
	return a % b
}

//go:noinline
func mul_int16_ssa(a, b int16) int16 {
	return a * b
}

//go:noinline
func add_uint8_ssa(a, b uint8) uint8 {
	return a + b
}

//go:noinline
func sub_uint8_ssa(a, b uint8) uint8 {
	return a - b
}

//go:noinline
func div_uint8_ssa(a, b uint8) uint8 {
	return a / b
}

//go:noinline
func mod_uint8_ssa(a, b uint8) uint8 {
	return a % b
}

//go:noinline
func mul_uint8_ssa(a, b uint8) uint8 {
	return a * b
}

//go:noinline
func add_int8_ssa(a, b int8) int8 {
	return a + b
}

//go:noinline
func sub_int8_ssa(a, b int8) int8 {
	return a - b
}

//go:noinline
func div_int8_ssa(a, b int8) int8 {
	return a / b
}

//go:noinline
func mod_int8_ssa(a, b int8) int8 {
	return a % b
}

//go:noinline
func mul_int8_ssa(a, b int8) int8 {
	return a * b
}

var uint64_data []utd64 = []utd64{utd64{a: 0, b: 0, add: 0, sub: 0, mul: 0},
	utd64{a: 0, b: 1, add: 1, sub: 18446744073709551615, mul: 0, div: 0, mod: 0},
	utd64{a: 0, b: 4294967296, add: 4294967296, sub: 18446744069414584320, mul: 0, div: 0, mod: 0},
	utd64{a: 0, b: 18446744073709551615, add: 18446744073709551615, sub: 1, mul: 0, div: 0, mod: 0},
	utd64{a: 1, b: 0, add: 1, sub: 1, mul: 0},
	utd64{a: 1, b: 1, add: 2, sub: 0, mul: 1, div: 1, mod: 0},
	utd64{a: 1, b: 4294967296, add: 4294967297, sub: 18446744069414584321, mul: 4294967296, div: 0, mod: 1},
	utd64{a: 1, b: 18446744073709551615, add: 0, sub: 2, mul: 18446744073709551615, div: 0, mod: 1},
	utd64{a: 4294967296, b: 0, add: 4294967296, sub: 4294967296, mul: 0},
	utd64{a: 4294967296, b: 1, add: 4294967297, sub: 4294967295, mul: 4294967296, div: 4294967296, mod: 0},
	utd64{a: 4294967296, b: 4294967296, add: 8589934592, sub: 0, mul: 0, div: 1, mod: 0},
	utd64{a: 4294967296, b: 18446744073709551615, add: 4294967295, sub: 4294967297, mul: 18446744069414584320, div: 0, mod: 4294967296},
	utd64{a: 18446744073709551615, b: 0, add: 18446744073709551615, sub: 18446744073709551615, mul: 0},
	utd64{a: 18446744073709551615, b: 1, add: 0, sub: 18446744073709551614, mul: 18446744073709551615, div: 18446744073709551615, mod: 0},
	utd64{a: 18446744073709551615, b: 4294967296, add: 4294967295, sub: 18446744069414584319, mul: 18446744069414584320, div: 4294967295, mod: 4294967295},
	utd64{a: 18446744073709551615, b: 18446744073709551615, add: 18446744073709551614, sub: 0, mul: 1, div: 1, mod: 0},
}
var int64_data []itd64 = []itd64{itd64{a: -9223372036854775808, b: -9223372036854775808, add: 0, sub: 0, mul: 0, div: 1, mod: 0},
	itd64{a: -9223372036854775808, b: -9223372036854775807, add: 1, sub: -1, mul: -9223372036854775808, div: 1, mod: -1},
	itd64{a: -9223372036854775808, b: -4294967296, add: 9223372032559808512, sub: -9223372032559808512, mul: 0, div: 2147483648, mod: 0},
	itd64{a: -9223372036854775808, b: -1, add: 9223372036854775807, sub: -9223372036854775807, mul: -9223372036854775808, div: -9223372036854775808, mod: 0},
	itd64{a: -9223372036854775808, b: 0, add: -9223372036854775808, sub: -9223372036854775808, mul: 0},
	itd64{a: -9223372036854775808, b: 1, add: -9223372036854775807, sub: 9223372036854775807, mul: -9223372036854775808, div: -9223372036854775808, mod: 0},
	itd64{a: -9223372036854775808, b: 4294967296, add: -9223372032559808512, sub: 9223372032559808512, mul: 0, div: -2147483648, mod: 0},
	itd64{a: -9223372036854775808, b: 9223372036854775806, add: -2, sub: 2, mul: 0, div: -1, mod: -2},
	itd64{a: -9223372036854775808, b: 9223372036854775807, add: -1, sub: 1, mul: -9223372036854775808, div: -1, mod: -1},
	itd64{a: -9223372036854775807, b: -9223372036854775808, add: 1, sub: 1, mul: -9223372036854775808, div: 0, mod: -9223372036854775807},
	itd64{a: -9223372036854775807, b: -9223372036854775807, add: 2, sub: 0, mul: 1, div: 1, mod: 0},
	itd64{a: -9223372036854775807, b: -4294967296, add: 9223372032559808513, sub: -9223372032559808511, mul: -4294967296, div: 2147483647, mod: -4294967295},
	itd64{a: -9223372036854775807, b: -1, add: -9223372036854775808, sub: -9223372036854775806, mul: 9223372036854775807, div: 9223372036854775807, mod: 0},
	itd64{a: -9223372036854775807, b: 0, add: -9223372036854775807, sub: -9223372036854775807, mul: 0},
	itd64{a: -9223372036854775807, b: 1, add: -9223372036854775806, sub: -9223372036854775808, mul: -9223372036854775807, div: -9223372036854775807, mod: 0},
	itd64{a: -9223372036854775807, b: 4294967296, add: -9223372032559808511, sub: 9223372032559808513, mul: 4294967296, div: -2147483647, mod: -4294967295},
	itd64{a: -9223372036854775807, b: 9223372036854775806, add: -1, sub: 3, mul: 9223372036854775806, div: -1, mod: -1},
	itd64{a: -9223372036854775807, b: 9223372036854775807, add: 0, sub: 2, mul: -1, div: -1, mod: 0},
	itd64{a: -4294967296, b: -9223372036854775808, add: 9223372032559808512, sub: 9223372032559808512, mul: 0, div: 0, mod: -4294967296},
	itd64{a: -4294967296, b: -9223372036854775807, add: 9223372032559808513, sub: 9223372032559808511, mul: -4294967296, div: 0, mod: -4294967296},
	itd64{a: -4294967296, b: -4294967296, add: -8589934592, sub: 0, mul: 0, div: 1, mod: 0},
	itd64{a: -4294967296, b: -1, add: -4294967297, sub: -4294967295, mul: 4294967296, div: 4294967296, mod: 0},
	itd64{a: -4294967296, b: 0, add: -4294967296, sub: -4294967296, mul: 0},
	itd64{a: -4294967296, b: 1, add: -4294967295, sub: -4294967297, mul: -4294967296, div: -4294967296, mod: 0},
	itd64{a: -4294967296, b: 4294967296, add: 0, sub: -8589934592, mul: 0, div: -1, mod: 0},
	itd64{a: -4294967296, b: 9223372036854775806, add: 9223372032559808510, sub: 9223372032559808514, mul: 8589934592, div: 0, mod: -4294967296},
	itd64{a: -4294967296, b: 9223372036854775807, add: 9223372032559808511, sub: 9223372032559808513, mul: 4294967296, div: 0, mod: -4294967296},
	itd64{a: -1, b: -9223372036854775808, add: 9223372036854775807, sub: 9223372036854775807, mul: -9223372036854775808, div: 0, mod: -1},
	itd64{a: -1, b: -9223372036854775807, add: -9223372036854775808, sub: 9223372036854775806, mul: 9223372036854775807, div: 0, mod: -1},
	itd64{a: -1, b: -4294967296, add: -4294967297, sub: 4294967295, mul: 4294967296, div: 0, mod: -1},
	itd64{a: -1, b: -1, add: -2, sub: 0, mul: 1, div: 1, mod: 0},
	itd64{a: -1, b: 0, add: -1, sub: -1, mul: 0},
	itd64{a: -1, b: 1, add: 0, sub: -2, mul: -1, div: -1, mod: 0},
	itd64{a: -1, b: 4294967296, add: 4294967295, sub: -4294967297, mul: -4294967296, div: 0, mod: -1},
	itd64{a: -1, b: 9223372036854775806, add: 9223372036854775805, sub: -9223372036854775807, mul: -9223372036854775806, div: 0, mod: -1},
	itd64{a: -1, b: 9223372036854775807, add: 9223372036854775806, sub: -9223372036854775808, mul: -9223372036854775807, div: 0, mod: -1},
	itd64{a: 0, b: -9223372036854775808, add: -9223372036854775808, sub: -9223372036854775808, mul: 0, div: 0, mod: 0},
	itd64{a: 0, b: -9223372036854775807, add: -9223372036854775807, sub: 9223372036854775807, mul: 0, div: 0, mod: 0},
	itd64{a: 0, b: -4294967296, add: -4294967296, sub: 4294967296, mul: 0, div: 0, mod: 0},
	itd64{a: 0, b: -1, add: -1, sub: 1, mul: 0, div: 0, mod: 0},
	itd64{a: 0, b: 0, add: 0, sub: 0, mul: 0},
	itd64{a: 0, b: 1, add: 1, sub: -1, mul: 0, div: 0, mod: 0},
	itd64{a: 0, b: 4294967296, add: 4294967296, sub: -4294967296, mul: 0, div: 0, mod: 0},
	itd64{a: 0, b: 9223372036854775806, add: 9223372036854775806, sub: -9223372036854775806, mul: 0, div: 0, mod: 0},
	itd64{a: 0, b: 9223372036854775807, add: 9223372036854775807, sub: -9223372036854775807, mul: 0, div: 0, mod: 0},
	itd64{a: 1, b: -9223372036854775808, add: -9223372036854775807, sub: -9223372036854775807, mul: -9223372036854775808, div: 0, mod: 1},
	itd64{a: 1, b: -9223372036854775807, add: -9223372036854775806, sub: -9223372036854775808, mul: -9223372036854775807, div: 0, mod: 1},
	itd64{a: 1, b: -4294967296, add: -4294967295, sub: 4294967297, mul: -4294967296, div: 0, mod: 1},
	itd64{a: 1, b: -1, add: 0, sub: 2, mul: -1, div: -1, mod: 0},
	itd64{a: 1, b: 0, add: 1, sub: 1, mul: 0},
	itd64{a: 1, b: 1, add: 2, sub: 0, mul: 1, div: 1, mod: 0},
	itd64{a: 1, b: 4294967296, add: 4294967297, sub: -4294967295, mul: 4294967296, div: 0, mod: 1},
	itd64{a: 1, b: 9223372036854775806, add: 9223372036854775807, sub: -9223372036854775805, mul: 9223372036854775806, div: 0, mod: 1},
	itd64{a: 1, b: 9223372036854775807, add: -9223372036854775808, sub: -9223372036854775806, mul: 9223372036854775807, div: 0, mod: 1},
	itd64{a: 4294967296, b: -9223372036854775808, add: -9223372032559808512, sub: -9223372032559808512, mul: 0, div: 0, mod: 4294967296},
	itd64{a: 4294967296, b: -9223372036854775807, add: -9223372032559808511, sub: -9223372032559808513, mul: 4294967296, div: 0, mod: 4294967296},
	itd64{a: 4294967296, b: -4294967296, add: 0, sub: 8589934592, mul: 0, div: -1, mod: 0},
	itd64{a: 4294967296, b: -1, add: 4294967295, sub: 4294967297, mul: -4294967296, div: -4294967296, mod: 0},
	itd64{a: 4294967296, b: 0, add: 4294967296, sub: 4294967296, mul: 0},
	itd64{a: 4294967296, b: 1, add: 4294967297, sub: 4294967295, mul: 4294967296, div: 4294967296, mod: 0},
	itd64{a: 4294967296, b: 4294967296, add: 8589934592, sub: 0, mul: 0, div: 1, mod: 0},
	itd64{a: 4294967296, b: 9223372036854775806, add: -9223372032559808514, sub: -9223372032559808510, mul: -8589934592, div: 0, mod: 4294967296},
	itd64{a: 4294967296, b: 9223372036854775807, add: -9223372032559808513, sub: -9223372032559808511, mul: -4294967296, div: 0, mod: 4294967296},
	itd64{a: 9223372036854775806, b: -9223372036854775808, add: -2, sub: -2, mul: 0, div: 0, mod: 9223372036854775806},
	itd64{a: 9223372036854775806, b: -9223372036854775807, add: -1, sub: -3, mul: 9223372036854775806, div: 0, mod: 9223372036854775806},
	itd64{a: 9223372036854775806, b: -4294967296, add: 9223372032559808510, sub: -9223372032559808514, mul: 8589934592, div: -2147483647, mod: 4294967294},
	itd64{a: 9223372036854775806, b: -1, add: 9223372036854775805, sub: 9223372036854775807, mul: -9223372036854775806, div: -9223372036854775806, mod: 0},
	itd64{a: 9223372036854775806, b: 0, add: 9223372036854775806, sub: 9223372036854775806, mul: 0},
	itd64{a: 9223372036854775806, b: 1, add: 9223372036854775807, sub: 9223372036854775805, mul: 9223372036854775806, div: 9223372036854775806, mod: 0},
	itd64{a: 9223372036854775806, b: 4294967296, add: -9223372032559808514, sub: 9223372032559808510, mul: -8589934592, div: 2147483647, mod: 4294967294},
	itd64{a: 9223372036854775806, b: 9223372036854775806, add: -4, sub: 0, mul: 4, div: 1, mod: 0},
	itd64{a: 9223372036854775806, b: 9223372036854775807, add: -3, sub: -1, mul: -9223372036854775806, div: 0, mod: 9223372036854775806},
	itd64{a: 9223372036854775807, b: -9223372036854775808, add: -1, sub: -1, mul: -9223372036854775808, div: 0, mod: 9223372036854775807},
	itd64{a: 9223372036854775807, b: -9223372036854775807, add: 0, sub: -2, mul: -1, div: -1, mod: 0},
	itd64{a: 9223372036854775807, b: -4294967296, add: 9223372032559808511, sub: -9223372032559808513, mul: 4294967296, div: -2147483647, mod: 4294967295},
	itd64{a: 9223372036854775807, b: -1, add: 9223372036854775806, sub: -9223372036854775808, mul: -9223372036854775807, div: -9223372036854775807, mod: 0},
	itd64{a: 9223372036854775807, b: 0, add: 9223372036854775807, sub: 9223372036854775807, mul: 0},
	itd64{a: 9223372036854775807, b: 1, add: -9223372036854775808, sub: 9223372036854775806, mul: 9223372036854775807, div: 9223372036854775807, mod: 0},
	itd64{a: 9223372036854775807, b: 4294967296, add: -9223372032559808513, sub: 9223372032559808511, mul: -4294967296, div: 2147483647, mod: 4294967295},
	itd64{a: 9223372036854775807, b: 9223372036854775806, add: -3, sub: 1, mul: -9223372036854775806, div: 1, mod: 1},
	itd64{a: 9223372036854775807, b: 9223372036854775807, add: -2, sub: 0, mul: 1, div: 1, mod: 0},
}
var uint32_data []utd32 = []utd32{utd32{a: 0, b: 0, add: 0, sub: 0, mul: 0},
	utd32{a: 0, b: 1, add: 1, sub: 4294967295, mul: 0, div: 0, mod: 0},
	utd32{a: 0, b: 4294967295, add: 4294967295, sub: 1, mul: 0, div: 0, mod: 0},
	utd32{a: 1, b: 0, add: 1, sub: 1, mul: 0},
	utd32{a: 1, b: 1, add: 2, sub: 0, mul: 1, div: 1, mod: 0},
	utd32{a: 1, b: 4294967295, add: 0, sub: 2, mul: 4294967295, div: 0, mod: 1},
	utd32{a: 4294967295, b: 0, add: 4294967295, sub: 4294967295, mul: 0},
	utd32{a: 4294967295, b: 1, add: 0, sub: 4294967294, mul: 4294967295, div: 4294967295, mod: 0},
	utd32{a: 4294967295, b: 4294967295, add: 4294967294, sub: 0, mul: 1, div: 1, mod: 0},
}
var int32_data []itd32 = []itd32{itd32{a: -2147483648, b: -2147483648, add: 0, sub: 0, mul: 0, div: 1, mod: 0},
	itd32{a: -2147483648, b: -2147483647, add: 1, sub: -1, mul: -2147483648, div: 1, mod: -1},
	itd32{a: -2147483648, b: -1, add: 2147483647, sub: -2147483647, mul: -2147483648, div: -2147483648, mod: 0},
	itd32{a: -2147483648, b: 0, add: -2147483648, sub: -2147483648, mul: 0},
	itd32{a: -2147483648, b: 1, add: -2147483647, sub: 2147483647, mul: -2147483648, div: -2147483648, mod: 0},
	itd32{a: -2147483648, b: 2147483647, add: -1, sub: 1, mul: -2147483648, div: -1, mod: -1},
	itd32{a: -2147483647, b: -2147483648, add: 1, sub: 1, mul: -2147483648, div: 0, mod: -2147483647},
	itd32{a: -2147483647, b: -2147483647, add: 2, sub: 0, mul: 1, div: 1, mod: 0},
	itd32{a: -2147483647, b: -1, add: -2147483648, sub: -2147483646, mul: 2147483647, div: 2147483647, mod: 0},
	itd32{a: -2147483647, b: 0, add: -2147483647, sub: -2147483647, mul: 0},
	itd32{a: -2147483647, b: 1, add: -2147483646, sub: -2147483648, mul: -2147483647, div: -2147483647, mod: 0},
	itd32{a: -2147483647, b: 2147483647, add: 0, sub: 2, mul: -1, div: -1, mod: 0},
	itd32{a: -1, b: -2147483648, add: 2147483647, sub: 2147483647, mul: -2147483648, div: 0, mod: -1},
	itd32{a: -1, b: -2147483647, add: -2147483648, sub: 2147483646, mul: 2147483647, div: 0, mod: -1},
	itd32{a: -1, b: -1, add: -2, sub: 0, mul: 1, div: 1, mod: 0},
	itd32{a: -1, b: 0, add: -1, sub: -1, mul: 0},
	itd32{a: -1, b: 1, add: 0, sub: -2, mul: -1, div: -1, mod: 0},
	itd32{a: -1, b: 2147483647, add: 2147483646, sub: -2147483648, mul: -2147483647, div: 0, mod: -1},
	itd32{a: 0, b: -2147483648, add: -2147483648, sub: -2147483648, mul: 0, div: 0, mod: 0},
	itd32{a: 0, b: -2147483647, add: -2147483647, sub: 2147483647, mul: 0, div: 0, mod: 0},
	itd32{a: 0, b: -1, add: -1, sub: 1, mul: 0, div: 0, mod: 0},
	itd32{a: 0, b: 0, add: 0, sub: 0, mul: 0},
	itd32{a: 0, b: 1, add: 1, sub: -1, mul: 0, div: 0, mod: 0},
	itd32{a: 0, b: 2147483647, add: 2147483647, sub: -2147483647, mul: 0, div: 0, mod: 0},
	itd32{a: 1, b: -2147483648, add: -2147483647, sub: -2147483647, mul: -2147483648, div: 0, mod: 1},
	itd32{a: 1, b: -2147483647, add: -2147483646, sub: -2147483648, mul: -2147483647, div: 0, mod: 1},
	itd32{a: 1, b: -1, add: 0, sub: 2, mul: -1, div: -1, mod: 0},
	itd32{a: 1, b: 0, add: 1, sub: 1, mul: 0},
	itd32{a: 1, b: 1, add: 2, sub: 0, mul: 1, div: 1, mod: 0},
	itd32{a: 1, b: 2147483647, add: -2147483648, sub: -2147483646, mul: 2147483647, div: 0, mod: 1},
	itd32{a: 2147483647, b: -2147483648, add: -1, sub: -1, mul: -2147483648, div: 0, mod: 2147483647},
	itd32{a: 2147483647, b: -2147483647, add: 0, sub: -2, mul: -1, div: -1, mod: 0},
	itd32{a: 2147483647, b: -1, add: 2147483646, sub: -2147483648, mul: -2147483647, div: -2147483647, mod: 0},
	itd32{a: 2147483647, b: 0, add: 2147483647, sub: 2147483647, mul: 0},
	itd32{a: 2147483647, b: 1, add: -2147483648, sub: 2147483646, mul: 2147483647, div: 2147483647, mod: 0},
	itd32{a: 2147483647, b: 2147483647, add: -2, sub: 0, mul: 1, div: 1, mod: 0},
}
var uint16_data []utd16 = []utd16{utd16{a: 0, b: 0, add: 0, sub: 0, mul: 0},
	utd16{a: 0, b: 1, add: 1, sub: 65535, mul: 0, div: 0, mod: 0},
	utd16{a: 0, b: 65535, add: 65535, sub: 1, mul: 0, div: 0, mod: 0},
	utd16{a: 1, b: 0, add: 1, sub: 1, mul: 0},
	utd16{a: 1, b: 1, add: 2, sub: 0, mul: 1, div: 1, mod: 0},
	utd16{a: 1, b: 65535, add: 0, sub: 2, mul: 65535, div: 0, mod: 1},
	utd16{a: 65535, b: 0, add: 65535, sub: 65535, mul: 0},
	utd16{a: 65535, b: 1, add: 0, sub: 65534, mul: 65535, div: 65535, mod: 0},
	utd16{a: 65535, b: 65535, add: 65534, sub: 0, mul: 1, div: 1, mod: 0},
}
var int16_data []itd16 = []itd16{itd16{a: -32768, b: -32768, add: 0, sub: 0, mul: 0, div: 1, mod: 0},
	itd16{a: -32768, b: -32767, add: 1, sub: -1, mul: -32768, div: 1, mod: -1},
	itd16{a: -32768, b: -1, add: 32767, sub: -32767, mul: -32768, div: -32768, mod: 0},
	itd16{a: -32768, b: 0, add: -32768, sub: -32768, mul: 0},
	itd16{a: -32768, b: 1, add: -32767, sub: 32767, mul: -32768, div: -32768, mod: 0},
	itd16{a: -32768, b: 32766, add: -2, sub: 2, mul: 0, div: -1, mod: -2},
	itd16{a: -32768, b: 32767, add: -1, sub: 1, mul: -32768, div: -1, mod: -1},
	itd16{a: -32767, b: -32768, add: 1, sub: 1, mul: -32768, div: 0, mod: -32767},
	itd16{a: -32767, b: -32767, add: 2, sub: 0, mul: 1, div: 1, mod: 0},
	itd16{a: -32767, b: -1, add: -32768, sub: -32766, mul: 32767, div: 32767, mod: 0},
	itd16{a: -32767, b: 0, add: -32767, sub: -32767, mul: 0},
	itd16{a: -32767, b: 1, add: -32766, sub: -32768, mul: -32767, div: -32767, mod: 0},
	itd16{a: -32767, b: 32766, add: -1, sub: 3, mul: 32766, div: -1, mod: -1},
	itd16{a: -32767, b: 32767, add: 0, sub: 2, mul: -1, div: -1, mod: 0},
	itd16{a: -1, b: -32768, add: 32767, sub: 32767, mul: -32768, div: 0, mod: -1},
	itd16{a: -1, b: -32767, add: -32768, sub: 32766, mul: 32767, div: 0, mod: -1},
	itd16{a: -1, b: -1, add: -2, sub: 0, mul: 1, div: 1, mod: 0},
	itd16{a: -1, b: 0, add: -1, sub: -1, mul: 0},
	itd16{a: -1, b: 1, add: 0, sub: -2, mul: -1, div: -1, mod: 0},
	itd16{a: -1, b: 32766, add: 32765, sub: -32767, mul: -32766, div: 0, mod: -1},
	itd16{a: -1, b: 32767, add: 32766, sub: -32768, mul: -32767, div: 0, mod: -1},
	itd16{a: 0, b: -32768, add: -32768, sub: -32768, mul: 0, div: 0, mod: 0},
	itd16{a: 0, b: -32767, add: -32767, sub: 32767, mul: 0, div: 0, mod: 0},
	itd16{a: 0, b: -1, add: -1, sub: 1, mul: 0, div: 0, mod: 0},
	itd16{a: 0, b: 0, add: 0, sub: 0, mul: 0},
	itd16{a: 0, b: 1, add: 1, sub: -1, mul: 0, div: 0, mod: 0},
	itd16{a: 0, b: 32766, add: 32766, sub: -32766, mul: 0, div: 0, mod: 0},
	itd16{a: 0, b: 32767, add: 32767, sub: -32767, mul: 0, div: 0, mod: 0},
	itd16{a: 1, b: -32768, add: -32767, sub: -32767, mul: -32768, div: 0, mod: 1},
	itd16{a: 1, b: -32767, add: -32766, sub: -32768, mul: -32767, div: 0, mod: 1},
	itd16{a: 1, b: -1, add: 0, sub: 2, mul: -1, div: -1, mod: 0},
	itd16{a: 1, b: 0, add: 1, sub: 1, mul: 0},
	itd16{a: 1, b: 1, add: 2, sub: 0, mul: 1, div: 1, mod: 0},
	itd16{a: 1, b: 32766, add: 32767, sub: -32765, mul: 32766, div: 0, mod: 1},
	itd16{a: 1, b: 32767, add: -32768, sub: -32766, mul: 32767, div: 0, mod: 1},
	itd16{a: 32766, b: -32768, add: -2, sub: -2, mul: 0, div: 0, mod: 32766},
	itd16{a: 32766, b: -32767, add: -1, sub: -3, mul: 32766, div: 0, mod: 32766},
	itd16{a: 32766, b: -1, add: 32765, sub: 32767, mul: -32766, div: -32766, mod: 0},
	itd16{a: 32766, b: 0, add: 32766, sub: 32766, mul: 0},
	itd16{a: 32766, b: 1, add: 32767, sub: 32765, mul: 32766, div: 32766, mod: 0},
	itd16{a: 32766, b: 32766, add: -4, sub: 0, mul: 4, div: 1, mod: 0},
	itd16{a: 32766, b: 32767, add: -3, sub: -1, mul: -32766, div: 0, mod: 32766},
	itd16{a: 32767, b: -32768, add: -1, sub: -1, mul: -32768, div: 0, mod: 32767},
	itd16{a: 32767, b: -32767, add: 0, sub: -2, mul: -1, div: -1, mod: 0},
	itd16{a: 32767, b: -1, add: 32766, sub: -32768, mul: -32767, div: -32767, mod: 0},
	itd16{a: 32767, b: 0, add: 32767, sub: 32767, mul: 0},
	itd16{a: 32767, b: 1, add: -32768, sub: 32766, mul: 32767, div: 32767, mod: 0},
	itd16{a: 32767, b: 32766, add: -3, sub: 1, mul: -32766, div: 1, mod: 1},
	itd16{a: 32767, b: 32767, add: -2, sub: 0, mul: 1, div: 1, mod: 0},
}
var uint8_data []utd8 = []utd8{utd8{a: 0, b: 0, add: 0, sub: 0, mul: 0},
	utd8{a: 0, b: 1, add: 1, sub: 255, mul: 0, div: 0, mod: 0},
	utd8{a: 0, b: 255, add: 255, sub: 1, mul: 0, div: 0, mod: 0},
	utd8{a: 1, b: 0, add: 1, sub: 1, mul: 0},
	utd8{a: 1, b: 1, add: 2, sub: 0, mul: 1, div: 1, mod: 0},
	utd8{a: 1, b: 255, add: 0, sub: 2, mul: 255, div: 0, mod: 1},
	utd8{a: 255, b: 0, add: 255, sub: 255, mul: 0},
	utd8{a: 255, b: 1, add: 0, sub: 254, mul: 255, div: 255, mod: 0},
	utd8{a: 255, b: 255, add: 254, sub: 0, mul: 1, div: 1, mod: 0},
}
var int8_data []itd8 = []itd8{itd8{a: -128, b: -128, add: 0, sub: 0, mul: 0, div: 1, mod: 0},
	itd8{a: -128, b: -127, add: 1, sub: -1, mul: -128, div: 1, mod: -1},
	itd8{a: -128, b: -1, add: 127, sub: -127, mul: -128, div: -128, mod: 0},
	itd8{a: -128, b: 0, add: -128, sub: -128, mul: 0},
	itd8{a: -128, b: 1, add: -127, sub: 127, mul: -128, div: -128, mod: 0},
	itd8{a: -128, b: 126, add: -2, sub: 2, mul: 0, div: -1, mod: -2},
	itd8{a: -128, b: 127, add: -1, sub: 1, mul: -128, div: -1, mod: -1},
	itd8{a: -127, b: -128, add: 1, sub: 1, mul: -128, div: 0, mod: -127},
	itd8{a: -127, b: -127, add: 2, sub: 0, mul: 1, div: 1, mod: 0},
	itd8{a: -127, b: -1, add: -128, sub: -126, mul: 127, div: 127, mod: 0},
	itd8{a: -127, b: 0, add: -127, sub: -127, mul: 0},
	itd8{a: -127, b: 1, add: -126, sub: -128, mul: -127, div: -127, mod: 0},
	itd8{a: -127, b: 126, add: -1, sub: 3, mul: 126, div: -1, mod: -1},
	itd8{a: -127, b: 127, add: 0, sub: 2, mul: -1, div: -1, mod: 0},
	itd8{a: -1, b: -128, add: 127, sub: 127, mul: -128, div: 0, mod: -1},
	itd8{a: -1, b: -127, add: -128, sub: 126, mul: 127, div: 0, mod: -1},
	itd8{a: -1, b: -1, add: -2, sub: 0, mul: 1, div: 1, mod: 0},
	itd8{a: -1, b: 0, add: -1, sub: -1, mul: 0},
	itd8{a: -1, b: 1, add: 0, sub: -2, mul: -1, div: -1, mod: 0},
	itd8{a: -1, b: 126, add: 125, sub: -127, mul: -126, div: 0, mod: -1},
	itd8{a: -1, b: 127, add: 126, sub: -128, mul: -127, div: 0, mod: -1},
	itd8{a: 0, b: -128, add: -128, sub: -128, mul: 0, div: 0, mod: 0},
	itd8{a: 0, b: -127, add: -127, sub: 127, mul: 0, div: 0, mod: 0},
	itd8{a: 0, b: -1, add: -1, sub: 1, mul: 0, div: 0, mod: 0},
	itd8{a: 0, b: 0, add: 0, sub: 0, mul: 0},
	itd8{a: 0, b: 1, add: 1, sub: -1, mul: 0, div: 0, mod: 0},
	itd8{a: 0, b: 126, add: 126, sub: -126, mul: 0, div: 0, mod: 0},
	itd8{a: 0, b: 127, add: 127, sub: -127, mul: 0, div: 0, mod: 0},
	itd8{a: 1, b: -128, add: -127, sub: -127, mul: -128, div: 0, mod: 1},
	itd8{a: 1, b: -127, add: -126, sub: -128, mul: -127, div: 0, mod: 1},
	itd8{a: 1, b: -1, add: 0, sub: 2, mul: -1, div: -1, mod: 0},
	itd8{a: 1, b: 0, add: 1, sub: 1, mul: 0},
	itd8{a: 1, b: 1, add: 2, sub: 0, mul: 1, div: 1, mod: 0},
	itd8{a: 1, b: 126, add: 127, sub: -125, mul: 126, div: 0, mod: 1},
	itd8{a: 1, b: 127, add: -128, sub: -126, mul: 127, div: 0, mod: 1},
	itd8{a: 126, b: -128, add: -2, sub: -2, mul: 0, div: 0, mod: 126},
	itd8{a: 126, b: -127, add: -1, sub: -3, mul: 126, div: 0, mod: 126},
	itd8{a: 126, b: -1, add: 125, sub: 127, mul: -126, div: -126, mod: 0},
	itd8{a: 126, b: 0, add: 126, sub: 126, mul: 0},
	itd8{a: 126, b: 1, add: 127, sub: 125, mul: 126, div: 126, mod: 0},
	itd8{a: 126, b: 126, add: -4, sub: 0, mul: 4, div: 1, mod: 0},
	itd8{a: 126, b: 127, add: -3, sub: -1, mul: -126, div: 0, mod: 126},
	itd8{a: 127, b: -128, add: -1, sub: -1, mul: -128, div: 0, mod: 127},
	itd8{a: 127, b: -127, add: 0, sub: -2, mul: -1, div: -1, mod: 0},
	itd8{a: 127, b: -1, add: 126, sub: -128, mul: -127, div: -127, mod: 0},
	itd8{a: 127, b: 0, add: 127, sub: 127, mul: 0},
	itd8{a: 127, b: 1, add: -128, sub: 126, mul: 127, div: 127, mod: 0},
	itd8{a: 127, b: 126, add: -3, sub: 1, mul: -126, div: 1, mod: 1},
	itd8{a: 127, b: 127, add: -2, sub: 0, mul: 1, div: 1, mod: 0},
}
var failed bool

func main() {

	for _, v := range uint64_data {
		if got := add_uint64_ssa(v.a, v.b); got != v.add {
			fmt.Printf("add_uint64 %d+%d = %d, wanted %d\n", v.a, v.b, got, v.add)
			failed = true
		}
		if got := sub_uint64_ssa(v.a, v.b); got != v.sub {
			fmt.Printf("sub_uint64 %d-%d = %d, wanted %d\n", v.a, v.b, got, v.sub)
			failed = true
		}
		if v.b != 0 {
			if got := div_uint64_ssa(v.a, v.b); got != v.div {
				fmt.Printf("div_uint64 %d/%d = %d, wanted %d\n", v.a, v.b, got, v.div)
				failed = true
			}

		}
		if v.b != 0 {
			if got := mod_uint64_ssa(v.a, v.b); got != v.mod {
				fmt.Printf("mod_uint64 %d%%%d = %d, wanted %d\n", v.a, v.b, got, v.mod)
				failed = true
			}

		}
		if got := mul_uint64_ssa(v.a, v.b); got != v.mul {
			fmt.Printf("mul_uint64 %d*%d = %d, wanted %d\n", v.a, v.b, got, v.mul)
			failed = true
		}
	}
	for _, v := range int64_data {
		if got := add_int64_ssa(v.a, v.b); got != v.add {
			fmt.Printf("add_int64 %d+%d = %d, wanted %d\n", v.a, v.b, got, v.add)
			failed = true
		}
		if got := sub_int64_ssa(v.a, v.b); got != v.sub {
			fmt.Printf("sub_int64 %d-%d = %d, wanted %d\n", v.a, v.b, got, v.sub)
			failed = true
		}
		if v.b != 0 {
			if got := div_int64_ssa(v.a, v.b); got != v.div {
				fmt.Printf("div_int64 %d/%d = %d, wanted %d\n", v.a, v.b, got, v.div)
				failed = true
			}

		}
		if v.b != 0 {
			if got := mod_int64_ssa(v.a, v.b); got != v.mod {
				fmt.Printf("mod_int64 %d%%%d = %d, wanted %d\n", v.a, v.b, got, v.mod)
				failed = true
			}

		}
		if got := mul_int64_ssa(v.a, v.b); got != v.mul {
			fmt.Printf("mul_int64 %d*%d = %d, wanted %d\n", v.a, v.b, got, v.mul)
			failed = true
		}
	}
	for _, v := range uint32_data {
		if got := add_uint32_ssa(v.a, v.b); got != v.add {
			fmt.Printf("add_uint32 %d+%d = %d, wanted %d\n", v.a, v.b, got, v.add)
			failed = true
		}
		if got := sub_uint32_ssa(v.a, v.b); got != v.sub {
			fmt.Printf("sub_uint32 %d-%d = %d, wanted %d\n", v.a, v.b, got, v.sub)
			failed = true
		}
		if v.b != 0 {
			if got := div_uint32_ssa(v.a, v.b); got != v.div {
				fmt.Printf("div_uint32 %d/%d = %d, wanted %d\n", v.a, v.b, got, v.div)
				failed = true
			}

		}
		if v.b != 0 {
			if got := mod_uint32_ssa(v.a, v.b); got != v.mod {
				fmt.Printf("mod_uint32 %d%%%d = %d, wanted %d\n", v.a, v.b, got, v.mod)
				failed = true
			}

		}
		if got := mul_uint32_ssa(v.a, v.b); got != v.mul {
			fmt.Printf("mul_uint32 %d*%d = %d, wanted %d\n", v.a, v.b, got, v.mul)
			failed = true
		}
	}
	for _, v := range int32_data {
		if got := add_int32_ssa(v.a, v.b); got != v.add {
			fmt.Printf("add_int32 %d+%d = %d, wanted %d\n", v.a, v.b, got, v.add)
			failed = true
		}
		if got := sub_int32_ssa(v.a, v.b); got != v.sub {
			fmt.Printf("sub_int32 %d-%d = %d, wanted %d\n", v.a, v.b, got, v.sub)
			failed = true
		}
		if v.b != 0 {
			if got := div_int32_ssa(v.a, v.b); got != v.div {
				fmt.Printf("div_int32 %d/%d = %d, wanted %d\n", v.a, v.b, got, v.div)
				failed = true
			}

		}
		if v.b != 0 {
			if got := mod_int32_ssa(v.a, v.b); got != v.mod {
				fmt.Printf("mod_int32 %d%%%d = %d, wanted %d\n", v.a, v.b, got, v.mod)
				failed = true
			}

		}
		if got := mul_int32_ssa(v.a, v.b); got != v.mul {
			fmt.Printf("mul_int32 %d*%d = %d, wanted %d\n", v.a, v.b, got, v.mul)
			failed = true
		}
	}
	for _, v := range uint16_data {
		if got := add_uint16_ssa(v.a, v.b); got != v.add {
			fmt.Printf("add_uint16 %d+%d = %d, wanted %d\n", v.a, v.b, got, v.add)
			failed = true
		}
		if got := sub_uint16_ssa(v.a, v.b); got != v.sub {
			fmt.Printf("sub_uint16 %d-%d = %d, wanted %d\n", v.a, v.b, got, v.sub)
			failed = true
		}
		if v.b != 0 {
			if got := div_uint16_ssa(v.a, v.b); got != v.div {
				fmt.Printf("div_uint16 %d/%d = %d, wanted %d\n", v.a, v.b, got, v.div)
				failed = true
			}

		}
		if v.b != 0 {
			if got := mod_uint16_ssa(v.a, v.b); got != v.mod {
				fmt.Printf("mod_uint16 %d%%%d = %d, wanted %d\n", v.a, v.b, got, v.mod)
				failed = true
			}

		}
		if got := mul_uint16_ssa(v.a, v.b); got != v.mul {
			fmt.Printf("mul_uint16 %d*%d = %d, wanted %d\n", v.a, v.b, got, v.mul)
			failed = true
		}
	}
	for _, v := range int16_data {
		if got := add_int16_ssa(v.a, v.b); got != v.add {
			fmt.Printf("add_int16 %d+%d = %d, wanted %d\n", v.a, v.b, got, v.add)
			failed = true
		}
		if got := sub_int16_ssa(v.a, v.b); got != v.sub {
			fmt.Printf("sub_int16 %d-%d = %d, wanted %d\n", v.a, v.b, got, v.sub)
			failed = true
		}
		if v.b != 0 {
			if got := div_int16_ssa(v.a, v.b); got != v.div {
				fmt.Printf("div_int16 %d/%d = %d, wanted %d\n", v.a, v.b, got, v.div)
				failed = true
			}

		}
		if v.b != 0 {
			if got := mod_int16_ssa(v.a, v.b); got != v.mod {
				fmt.Printf("mod_int16 %d%%%d = %d, wanted %d\n", v.a, v.b, got, v.mod)
				failed = true
			}

		}
		if got := mul_int16_ssa(v.a, v.b); got != v.mul {
			fmt.Printf("mul_int16 %d*%d = %d, wanted %d\n", v.a, v.b, got, v.mul)
			failed = true
		}
	}
	for _, v := range uint8_data {
		if got := add_uint8_ssa(v.a, v.b); got != v.add {
			fmt.Printf("add_uint8 %d+%d = %d, wanted %d\n", v.a, v.b, got, v.add)
			failed = true
		}
		if got := sub_uint8_ssa(v.a, v.b); got != v.sub {
			fmt.Printf("sub_uint8 %d-%d = %d, wanted %d\n", v.a, v.b, got, v.sub)
			failed = true
		}
		if v.b != 0 {
			if got := div_uint8_ssa(v.a, v.b); got != v.div {
				fmt.Printf("div_uint8 %d/%d = %d, wanted %d\n", v.a, v.b, got, v.div)
				failed = true
			}

		}
		if v.b != 0 {
			if got := mod_uint8_ssa(v.a, v.b); got != v.mod {
				fmt.Printf("mod_uint8 %d%%%d = %d, wanted %d\n", v.a, v.b, got, v.mod)
				failed = true
			}

		}
		if got := mul_uint8_ssa(v.a, v.b); got != v.mul {
			fmt.Printf("mul_uint8 %d*%d = %d, wanted %d\n", v.a, v.b, got, v.mul)
			failed = true
		}
	}
	for _, v := range int8_data {
		if got := add_int8_ssa(v.a, v.b); got != v.add {
			fmt.Printf("add_int8 %d+%d = %d, wanted %d\n", v.a, v.b, got, v.add)
			failed = true
		}
		if got := sub_int8_ssa(v.a, v.b); got != v.sub {
			fmt.Printf("sub_int8 %d-%d = %d, wanted %d\n", v.a, v.b, got, v.sub)
			failed = true
		}
		if v.b != 0 {
			if got := div_int8_ssa(v.a, v.b); got != v.div {
				fmt.Printf("div_int8 %d/%d = %d, wanted %d\n", v.a, v.b, got, v.div)
				failed = true
			}

		}
		if v.b != 0 {
			if got := mod_int8_ssa(v.a, v.b); got != v.mod {
				fmt.Printf("mod_int8 %d%%%d = %d, wanted %d\n", v.a, v.b, got, v.mod)
				failed = true
			}

		}
		if got := mul_int8_ssa(v.a, v.b); got != v.mul {
			fmt.Printf("mul_int8 %d*%d = %d, wanted %d\n", v.a, v.b, got, v.mul)
			failed = true
		}
	}
	if failed {
		panic("tests failed")
	}
}
