// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && !(amd64 || wasm || arm64)

package simd

import (
	"fmt"
	"math"
	"math/bits"
)

// VectorSize returns the bit length of the emulated vector (fixed to 128).
func VectorBitSize() int {
	return 128
}

// Emulated returns whether simd is emulated.
func Emulated() bool {
	return true
}

// EmulatedCarrylessMultiply returns whether CarrylessMultiply is emulated.
// This sometimes matters to choice of algorithm (e.g., when computing CRC).
// The emulation's execution time does not depend on its inputs, so it is
// okay in that sense.
func EmulatedCarrylessMultiply() bool {
	return true
}

type _simd struct {
	_ [0]func(*_simd) *_simd
}

// Int8s represents a 128-bit vector of 16 int8 elements.
type Int8s struct {
	_    _simd
	a, b uint64
}

// LoadInt8s loads a slice of int8 into an Int8s vector.
func LoadInt8s(s []int8) Int8s {
	var a, b uint64
	for i := 0; i < 16; i++ {
		val := uint64(uint8(s[i]))
		if i < 8 {
			a |= val << (8 * i)
		} else {
			b |= val << (8 * (i - 8))
		}
	}
	return Int8s{a: a, b: b}
}

// LoadInt8sPart loads a partial slice of int8 into an Int8s vector.
func LoadInt8sPart(s []int8) (Int8s, int) {
	var a, b uint64
	n := len(s)
	if n > 16 {
		n = 16
	}
	for i := 0; i < n; i++ {
		val := uint64(uint8(s[i]))
		if i < 8 {
			a |= val << (8 * i)
		} else {
			b |= val << (8 * (i - 8))
		}
	}
	return Int8s{a: a, b: b}, n
}

func (x Int8s) get(i int) int8 {
	if i < 8 {
		return int8(x.a >> (8 * i))
	}
	return int8(x.b >> (8 * (i - 8)))
}

func (x *Int8s) set(i int, v int8) {
	val := uint64(uint8(v))
	if i < 8 {
		mask := uint64(0xff) << (8 * i)
		x.a = (x.a &^ mask) | (val << (8 * i))
	} else {
		mask := uint64(0xff) << (8 * (i - 8))
		x.b = (x.b &^ mask) | (val << (8 * (i - 8)))
	}
}

// Abs returns the element-wise absolute value of x.
func (x Int8s) Abs() Int8s {
	var res Int8s
	for i := 0; i < 16; i++ {
		v := x.get(i)
		if v < 0 {
			res.set(i, -v)
		} else {
			res.set(i, v)
		}
	}
	return res
}

// Add returns the element-wise sum of x and y.
func (x Int8s) Add(y Int8s) Int8s {
	var res Int8s
	for i := 0; i < 16; i++ {
		res.set(i, x.get(i)+y.get(i))
	}
	return res
}

// AddSaturated returns the element-wise saturated sum of x and y.
func (x Int8s) AddSaturated(y Int8s) Int8s {
	var res Int8s
	for i := 0; i < 16; i++ {
		sum := int(x.get(i)) + int(y.get(i))
		if sum > math.MaxInt8 {
			res.set(i, math.MaxInt8)
		} else if sum < math.MinInt8 {
			res.set(i, math.MinInt8)
		} else {
			res.set(i, int8(sum))
		}
	}
	return res
}

// And returns the bitwise AND of x and y.
func (x Int8s) And(y Int8s) Int8s {
	return Int8s{a: x.a & y.a, b: x.b & y.b}
}

// AndNot returns the bitwise AND NOT of x and y.
func (x Int8s) AndNot(y Int8s) Int8s {
	return Int8s{a: x.a &^ y.a, b: x.b &^ y.b}
}

// Equal returns a mask indicating where x and y are equal.
func (x Int8s) Equal(y Int8s) Mask8s {
	var res Mask8s
	for i := 0; i < 16; i++ {
		if x.get(i) == y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// Greater returns a mask indicating where x is greater than y.
func (x Int8s) Greater(y Int8s) Mask8s {
	var res Mask8s
	for i := 0; i < 16; i++ {
		if x.get(i) > y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// GreaterEqual returns a mask indicating where x is greater than or equal to y.
func (x Int8s) GreaterEqual(y Int8s) Mask8s {
	var res Mask8s
	for i := 0; i < 16; i++ {
		if x.get(i) >= y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// Less returns a mask indicating where x is less than y.
func (x Int8s) Less(y Int8s) Mask8s {
	var res Mask8s
	for i := 0; i < 16; i++ {
		if x.get(i) < y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// LessEqual returns a mask indicating where x is less than or equal to y.
func (x Int8s) LessEqual(y Int8s) Mask8s {
	var res Mask8s
	for i := 0; i < 16; i++ {
		if x.get(i) <= y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// NotEqual returns a mask indicating where x and y are not equal.
func (x Int8s) NotEqual(y Int8s) Mask8s {
	var res Mask8s
	for i := 0; i < 16; i++ {
		if x.get(i) != y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// Len returns the number of elements in the vector.
func (x Int8s) Len() int {
	return 16
}

// Masked returns a new vector with elements from x where mask is true, and zero elsewhere.
func (x Int8s) Masked(mask Mask8s) Int8s {
	return Int8s{a: x.a & mask.a, b: x.b & mask.b}
}

// Max returns the element-wise maximum of x and y.
func (x Int8s) Max(y Int8s) Int8s {
	var res Int8s
	for i := 0; i < 16; i++ {
		vx := x.get(i)
		vy := y.get(i)
		if vx > vy {
			res.set(i, vx)
		} else {
			res.set(i, vy)
		}
	}
	return res
}

// Mul returns the element-wise product of x and y.
func (x Int8s) Mul(y Int8s) Int8s {
	var res Int8s
	for i := 0; i < 16; i++ {
		res.set(i, x.get(i)*y.get(i))
	}
	return res
}

// IfElse returns a new vector with elements from x where mask is true, and y where mask is false.
func (x Int8s) IfElse(mask Mask8s, y Int8s) Int8s {
	return Int8s{
		a: (x.a & mask.a) | (y.a &^ mask.a),
		b: (x.b & mask.b) | (y.b &^ mask.b),
	}
}

// Min returns the element-wise minimum of x and y.
func (x Int8s) Min(y Int8s) Int8s {
	var res Int8s
	for i := 0; i < 16; i++ {
		vx := x.get(i)
		vy := y.get(i)
		if vx < vy {
			res.set(i, vx)
		} else {
			res.set(i, vy)
		}
	}
	return res
}

// Neg returns the element-wise negation of x.
func (x Int8s) Neg() Int8s {
	var res Int8s
	for i := 0; i < 16; i++ {
		res.set(i, -x.get(i))
	}
	return res
}

// Not returns the bitwise NOT of x.
func (x Int8s) Not() Int8s {
	return Int8s{a: ^x.a, b: ^x.b}
}

// Or returns the bitwise OR of x and y.
func (x Int8s) Or(y Int8s) Int8s {
	return Int8s{a: x.a | y.a, b: x.b | y.b}
}

// Store stores the vector elements into the slice s.
func (x Int8s) Store(s []int8) {
	for i := 0; i < 16 && i < len(s); i++ {
		s[i] = x.get(i)
	}
}

// StorePart stores a partial vector into the slice s.
func (x Int8s) StorePart(s []int8) {
	x.Store(s)
}

// String returns a string representation of the vector.
func (x Int8s) String() string {
	var parts [16]int8
	for i := 0; i < 16; i++ {
		parts[i] = x.get(i)
	}
	return fmt.Sprint(parts)
}

// Sub returns the element-wise difference of x and y.
func (x Int8s) Sub(y Int8s) Int8s {
	var res Int8s
	for i := 0; i < 16; i++ {
		res.set(i, x.get(i)-y.get(i))
	}
	return res
}

// SubSaturated returns the element-wise saturated difference of x and y.
func (x Int8s) SubSaturated(y Int8s) Int8s {
	var res Int8s
	for i := 0; i < 16; i++ {
		diff := int(x.get(i)) - int(y.get(i))
		if diff > math.MaxInt8 {
			res.set(i, math.MaxInt8)
		} else if diff < math.MinInt8 {
			res.set(i, math.MinInt8)
		} else {
			res.set(i, int8(diff))
		}
	}
	return res
}

// ToMask returns a mask representation of the vector.
func (x Int8s) ToMask() Mask8s {
	var res Mask8s
	for i := 0; i < 16; i++ {
		if x.get(i) != 0 {
			res.set(i, true)
		}
	}
	return res
}

// Xor returns the bitwise XOR of x and y.
func (x Int8s) Xor(y Int8s) Int8s {
	return Int8s{a: x.a ^ y.a, b: x.b ^ y.b}
}

// ConvertToUint8 converts the vector elements to uint8.
func (x Int8s) ConvertToUint8() Uint8s {
	return Uint8s{a: x.a, b: x.b}
}

// ToBits reinterprets the vector bits as a Uint8s vector.
func (x Int8s) ToBits() Uint8s {
	return Uint8s{a: x.a, b: x.b}
}

// Int16s represents a 128-bit vector of 8 int16 elements.
type Int16s struct {
	_    _simd
	a, b uint64
}

// LoadInt16s loads a slice of int16 into an Int16s vector.
func LoadInt16s(s []int16) Int16s {
	var a, b uint64
	for i := 0; i < 8; i++ {
		val := uint64(uint16(s[i]))
		if i < 4 {
			a |= val << (16 * i)
		} else {
			b |= val << (16 * (i - 4))
		}
	}
	return Int16s{a: a, b: b}
}

// LoadInt16sPart loads a partial slice of int16 into an Int16s vector.
func LoadInt16sPart(s []int16) (Int16s, int) {
	var a, b uint64
	n := len(s)
	if n > 8 {
		n = 8
	}
	for i := 0; i < n; i++ {
		val := uint64(uint16(s[i]))
		if i < 4 {
			a |= val << (16 * i)
		} else {
			b |= val << (16 * (i - 4))
		}
	}
	return Int16s{a: a, b: b}, n
}

func (x Int16s) get(i int) int16 {
	if i < 4 {
		return int16(x.a >> (16 * i))
	}
	return int16(x.b >> (16 * (i - 4)))
}

func (x *Int16s) set(i int, v int16) {
	val := uint64(uint16(v))
	if i < 4 {
		mask := uint64(0xffff) << (16 * i)
		x.a = (x.a &^ mask) | (val << (16 * i))
	} else {
		mask := uint64(0xffff) << (16 * (i - 4))
		x.b = (x.b &^ mask) | (val << (16 * (i - 4)))
	}
}

// Abs returns the element-wise absolute value of x.
func (x Int16s) Abs() Int16s {
	var res Int16s
	for i := 0; i < 8; i++ {
		v := x.get(i)
		if v < 0 {
			res.set(i, -v)
		} else {
			res.set(i, v)
		}
	}
	return res
}

// Add returns the element-wise sum of x and y.
func (x Int16s) Add(y Int16s) Int16s {
	var res Int16s
	for i := 0; i < 8; i++ {
		res.set(i, x.get(i)+y.get(i))
	}
	return res
}

// AddSaturated returns the element-wise saturated sum of x and y.
func (x Int16s) AddSaturated(y Int16s) Int16s {
	var res Int16s
	for i := 0; i < 8; i++ {
		sum := int(x.get(i)) + int(y.get(i))
		if sum > math.MaxInt16 {
			res.set(i, math.MaxInt16)
		} else if sum < math.MinInt16 {
			res.set(i, math.MinInt16)
		} else {
			res.set(i, int16(sum))
		}
	}
	return res
}

// And returns the bitwise AND of x and y.
func (x Int16s) And(y Int16s) Int16s {
	return Int16s{a: x.a & y.a, b: x.b & y.b}
}

// AndNot returns the bitwise AND NOT of x and y.
func (x Int16s) AndNot(y Int16s) Int16s {
	return Int16s{a: x.a &^ y.a, b: x.b &^ y.b}
}

// Equal returns a mask indicating where x and y are equal.
func (x Int16s) Equal(y Int16s) Mask16s {
	var res Mask16s
	for i := 0; i < 8; i++ {
		if x.get(i) == y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// Greater returns a mask indicating where x is greater than y.
func (x Int16s) Greater(y Int16s) Mask16s {
	var res Mask16s
	for i := 0; i < 8; i++ {
		if x.get(i) > y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// GreaterEqual returns a mask indicating where x is greater than or equal to y.
func (x Int16s) GreaterEqual(y Int16s) Mask16s {
	var res Mask16s
	for i := 0; i < 8; i++ {
		if x.get(i) >= y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// Less returns a mask indicating where x is less than y.
func (x Int16s) Less(y Int16s) Mask16s {
	var res Mask16s
	for i := 0; i < 8; i++ {
		if x.get(i) < y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// LessEqual returns a mask indicating where x is less than or equal to y.
func (x Int16s) LessEqual(y Int16s) Mask16s {
	var res Mask16s
	for i := 0; i < 8; i++ {
		if x.get(i) <= y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// NotEqual returns a mask indicating where x and y are not equal.
func (x Int16s) NotEqual(y Int16s) Mask16s {
	var res Mask16s
	for i := 0; i < 8; i++ {
		if x.get(i) != y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// Len returns the number of elements in the vector.
func (x Int16s) Len() int {
	return 8
}

// Masked returns a new vector with elements from x where mask is true, and zero elsewhere.
func (x Int16s) Masked(mask Mask16s) Int16s {
	return Int16s{a: x.a & mask.a, b: x.b & mask.b}
}

// Max returns the element-wise maximum of x and y.
func (x Int16s) Max(y Int16s) Int16s {
	var res Int16s
	for i := 0; i < 8; i++ {
		vx := x.get(i)
		vy := y.get(i)
		if vx > vy {
			res.set(i, vx)
		} else {
			res.set(i, vy)
		}
	}
	return res
}

// IfElse returns a new vector with elements from x where mask is true, and y where mask is false.
func (x Int16s) IfElse(mask Mask16s, y Int16s) Int16s {
	return Int16s{
		a: (x.a & mask.a) | (y.a &^ mask.a),
		b: (x.b & mask.b) | (y.b &^ mask.b),
	}
}

// Min returns the element-wise minimum of x and y.
func (x Int16s) Min(y Int16s) Int16s {
	var res Int16s
	for i := 0; i < 8; i++ {
		vx := x.get(i)
		vy := y.get(i)
		if vx < vy {
			res.set(i, vx)
		} else {
			res.set(i, vy)
		}
	}
	return res
}

// Mul returns the element-wise product of x and y.
func (x Int16s) Mul(y Int16s) Int16s {
	var res Int16s
	for i := 0; i < 8; i++ {
		res.set(i, x.get(i)*y.get(i))
	}
	return res
}

// Neg returns the element-wise negation of x.
func (x Int16s) Neg() Int16s {
	var res Int16s
	for i := 0; i < 8; i++ {
		res.set(i, -x.get(i))
	}
	return res
}

// Not returns the bitwise NOT of x.
func (x Int16s) Not() Int16s {
	return Int16s{a: ^x.a, b: ^x.b}
}

// Or returns the bitwise OR of x and y.
func (x Int16s) Or(y Int16s) Int16s {
	return Int16s{a: x.a | y.a, b: x.b | y.b}
}

// ShiftAllLeft shifts all elements left by y bits.
func (x Int16s) ShiftAllLeft(y uint8) Int16s {
	var res Int16s
	for i := 0; i < 8; i++ {
		res.set(i, x.get(i)<<y)
	}
	return res
}

// ShiftAllRight shifts all elements right by y bits.
func (x Int16s) ShiftAllRight(y uint8) Int16s {
	var res Int16s
	for i := 0; i < 8; i++ {
		res.set(i, x.get(i)>>y)
	}
	return res
}

// RotateAllLeft rotates all elements left by dist bits.
func (x Int16s) RotateAllLeft(dist uint64) Int16s {
	var res Int16s
	d := dist & 15
	for i := 0; i < 8; i++ {
		u := uint16(x.get(i))
		r := (u << d) | (u >> ((16 - d) & 15))
		res.set(i, int16(r))
	}
	return res
}

// RotateAllRight rotates all elements right by dist bits.
func (x Int16s) RotateAllRight(dist uint64) Int16s {
	var res Int16s
	d := dist & 15
	for i := 0; i < 8; i++ {
		u := uint16(x.get(i))
		r := (u >> d) | (u << ((16 - d) & 15))
		res.set(i, int16(r))
	}
	return res
}

// Store stores the vector elements into the slice s.
func (x Int16s) Store(s []int16) {
	for i := 0; i < 8 && i < len(s); i++ {
		s[i] = x.get(i)
	}
}

// StorePart stores a partial vector into the slice s.
func (x Int16s) StorePart(s []int16) {
	x.Store(s)
}

// String returns a string representation of the vector.
func (x Int16s) String() string {
	var parts [8]int16
	for i := 0; i < 8; i++ {
		parts[i] = x.get(i)
	}
	return fmt.Sprint(parts)
}

// Sub returns the element-wise difference of x and y.
func (x Int16s) Sub(y Int16s) Int16s {
	var res Int16s
	for i := 0; i < 8; i++ {
		res.set(i, x.get(i)-y.get(i))
	}
	return res
}

// SubSaturated returns the element-wise saturated difference of x and y.
func (x Int16s) SubSaturated(y Int16s) Int16s {
	var res Int16s
	for i := 0; i < 8; i++ {
		diff := int(x.get(i)) - int(y.get(i))
		if diff > math.MaxInt16 {
			res.set(i, math.MaxInt16)
		} else if diff < math.MinInt16 {
			res.set(i, math.MinInt16)
		} else {
			res.set(i, int16(diff))
		}
	}
	return res
}

// ToMask returns a mask representation of the vector.
func (x Int16s) ToMask() Mask16s {
	var res Mask16s
	for i := 0; i < 8; i++ {
		if x.get(i) != 0 {
			res.set(i, true)
		}
	}
	return res
}

// Xor returns the bitwise XOR of x and y.
func (x Int16s) Xor(y Int16s) Int16s {
	return Int16s{a: x.a ^ y.a, b: x.b ^ y.b}
}

// ConvertToUint16 converts the vector elements to uint16.
func (x Int16s) ConvertToUint16() Uint16s {
	return Uint16s{a: x.a, b: x.b}
}

// ToBits reinterprets the vector bits as a Uint16s vector.
func (x Int16s) ToBits() Uint16s {
	return Uint16s{a: x.a, b: x.b}
}

// Int32s represents a 128-bit vector of 4 int32 elements.
type Int32s struct {
	_    _simd
	a, b uint64
}

// LoadInt32s loads a slice of int32 into an Int32s vector.
func LoadInt32s(s []int32) Int32s {
	var a, b uint64
	for i := 0; i < 4; i++ {
		val := uint64(uint32(s[i]))
		if i < 2 {
			a |= val << (32 * i)
		} else {
			b |= val << (32 * (i - 2))
		}
	}
	return Int32s{a: a, b: b}
}

// LoadInt32sPart loads a partial slice of int32 into an Int32s vector.
func LoadInt32sPart(s []int32) (Int32s, int) {
	var a, b uint64
	n := len(s)
	if n > 4 {
		n = 4
	}
	for i := 0; i < n; i++ {
		val := uint64(uint32(s[i]))
		if i < 2 {
			a |= val << (32 * i)
		} else {
			b |= val << (32 * (i - 2))
		}
	}
	return Int32s{a: a, b: b}, n
}

func (x Int32s) get(i int) int32 {
	if i < 2 {
		return int32(x.a >> (32 * i))
	}
	return int32(x.b >> (32 * (i - 2)))
}

func (x *Int32s) set(i int, v int32) {
	val := uint64(uint32(v))
	if i < 2 {
		mask := uint64(0xffffffff) << (32 * i)
		x.a = (x.a &^ mask) | (val << (32 * i))
	} else {
		mask := uint64(0xffffffff) << (32 * (i - 2))
		x.b = (x.b &^ mask) | (val << (32 * (i - 2)))
	}
}

// Abs returns the element-wise absolute value of x.
func (x Int32s) Abs() Int32s {
	var res Int32s
	for i := 0; i < 4; i++ {
		v := x.get(i)
		if v < 0 {
			res.set(i, -v)
		} else {
			res.set(i, v)
		}
	}
	return res
}

// Add returns the element-wise sum of x and y.
func (x Int32s) Add(y Int32s) Int32s {
	var res Int32s
	for i := 0; i < 4; i++ {
		res.set(i, x.get(i)+y.get(i))
	}
	return res
}

// And returns the bitwise AND of x and y.
func (x Int32s) And(y Int32s) Int32s {
	return Int32s{a: x.a & y.a, b: x.b & y.b}
}

// AndNot returns the bitwise AND NOT of x and y.
func (x Int32s) AndNot(y Int32s) Int32s {
	return Int32s{a: x.a &^ y.a, b: x.b &^ y.b}
}

// ConvertToFloat32 converts the vector elements to float32.
func (x Int32s) ConvertToFloat32() Float32s {
	var res Float32s
	for i := 0; i < 4; i++ {
		res.set(i, float32(x.get(i)))
	}
	return res
}

// Equal returns a mask indicating where x and y are equal.
func (x Int32s) Equal(y Int32s) Mask32s {
	var res Mask32s
	for i := 0; i < 4; i++ {
		if x.get(i) == y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// Greater returns a mask indicating where x is greater than y.
func (x Int32s) Greater(y Int32s) Mask32s {
	var res Mask32s
	for i := 0; i < 4; i++ {
		if x.get(i) > y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// GreaterEqual returns a mask indicating where x is greater than or equal to y.
func (x Int32s) GreaterEqual(y Int32s) Mask32s {
	var res Mask32s
	for i := 0; i < 4; i++ {
		if x.get(i) >= y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// Less returns a mask indicating where x is less than y.
func (x Int32s) Less(y Int32s) Mask32s {
	var res Mask32s
	for i := 0; i < 4; i++ {
		if x.get(i) < y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// LessEqual returns a mask indicating where x is less than or equal to y.
func (x Int32s) LessEqual(y Int32s) Mask32s {
	var res Mask32s
	for i := 0; i < 4; i++ {
		if x.get(i) <= y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// NotEqual returns a mask indicating where x and y are not equal.
func (x Int32s) NotEqual(y Int32s) Mask32s {
	var res Mask32s
	for i := 0; i < 4; i++ {
		if x.get(i) != y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// Len returns the number of elements in the vector.
func (x Int32s) Len() int {
	return 4
}

// Masked returns a new vector with elements from x where mask is true, and zero elsewhere.
func (x Int32s) Masked(mask Mask32s) Int32s {
	return Int32s{a: x.a & mask.a, b: x.b & mask.b}
}

// Max returns the element-wise maximum of x and y.
func (x Int32s) Max(y Int32s) Int32s {
	var res Int32s
	for i := 0; i < 4; i++ {
		vx := x.get(i)
		vy := y.get(i)
		if vx > vy {
			res.set(i, vx)
		} else {
			res.set(i, vy)
		}
	}
	return res
}

// IfElse returns a new vector with elements from x where mask is true, and y where mask is false.
func (x Int32s) IfElse(mask Mask32s, y Int32s) Int32s {
	return Int32s{
		a: (x.a & mask.a) | (y.a &^ mask.a),
		b: (x.b & mask.b) | (y.b &^ mask.b),
	}
}

// Min returns the element-wise minimum of x and y.
func (x Int32s) Min(y Int32s) Int32s {
	var res Int32s
	for i := 0; i < 4; i++ {
		vx := x.get(i)
		vy := y.get(i)
		if vx < vy {
			res.set(i, vx)
		} else {
			res.set(i, vy)
		}
	}
	return res
}

// Mul returns the element-wise product of x and y.
func (x Int32s) Mul(y Int32s) Int32s {
	var res Int32s
	for i := 0; i < 4; i++ {
		res.set(i, x.get(i)*y.get(i))
	}
	return res
}

// Neg returns the element-wise negation of x.
func (x Int32s) Neg() Int32s {
	var res Int32s
	for i := 0; i < 4; i++ {
		res.set(i, -x.get(i))
	}
	return res
}

// Not returns the bitwise NOT of x.
func (x Int32s) Not() Int32s {
	return Int32s{a: ^x.a, b: ^x.b}
}

// Or returns the bitwise OR of x and y.
func (x Int32s) Or(y Int32s) Int32s {
	return Int32s{a: x.a | y.a, b: x.b | y.b}
}

// ShiftAllLeft shifts all elements left by y bits.
func (x Int32s) ShiftAllLeft(y uint8) Int32s {
	var res Int32s
	for i := 0; i < 4; i++ {
		res.set(i, x.get(i)<<y)
	}
	return res
}

// ShiftAllRight shifts all elements right by y bits.
func (x Int32s) ShiftAllRight(y uint8) Int32s {
	var res Int32s
	for i := 0; i < 4; i++ {
		res.set(i, x.get(i)>>y)
	}
	return res
}

// RotateAllLeft rotates all elements left by dist bits.
func (x Int32s) RotateAllLeft(dist uint64) Int32s {
	var res Int32s
	d := dist & 31
	for i := 0; i < 4; i++ {
		u := uint32(x.get(i))
		r := (u << d) | (u >> ((32 - d) & 31))
		res.set(i, int32(r))
	}
	return res
}

// RotateAllRight rotates all elements right by dist bits.
func (x Int32s) RotateAllRight(dist uint64) Int32s {
	var res Int32s
	d := dist & 31
	for i := 0; i < 4; i++ {
		u := uint32(x.get(i))
		r := (u >> d) | (u << ((32 - d) & 31))
		res.set(i, int32(r))
	}
	return res
}

// Store stores the vector elements into the slice s.
func (x Int32s) Store(s []int32) {
	for i := 0; i < 4 && i < len(s); i++ {
		s[i] = x.get(i)
	}
}

// StorePart stores a partial vector into the slice s.
func (x Int32s) StorePart(s []int32) {
	x.Store(s)
}

// String returns a string representation of the vector.
func (x Int32s) String() string {
	var parts [4]int32
	for i := 0; i < 4; i++ {
		parts[i] = x.get(i)
	}
	return fmt.Sprint(parts)
}

// Sub returns the element-wise difference of x and y.
func (x Int32s) Sub(y Int32s) Int32s {
	var res Int32s
	for i := 0; i < 4; i++ {
		res.set(i, x.get(i)-y.get(i))
	}
	return res
}

// ToMask returns a mask representation of the vector.
func (x Int32s) ToMask() Mask32s {
	var res Mask32s
	for i := 0; i < 4; i++ {
		if x.get(i) != 0 {
			res.set(i, true)
		}
	}
	return res
}

// Xor returns the bitwise XOR of x and y.
func (x Int32s) Xor(y Int32s) Int32s {
	return Int32s{a: x.a ^ y.a, b: x.b ^ y.b}
}

// ConvertToUint32 converts the vector elements to uint32.
func (x Int32s) ConvertToUint32() Uint32s {
	return Uint32s{a: x.a, b: x.b}
}

// ToBits reinterprets the vector bits as a Uint32s vector.
func (x Int32s) ToBits() Uint32s {
	return Uint32s{a: x.a, b: x.b}
}

// Int64s represents a 128-bit vector of 2 int64 elements.
type Int64s struct {
	_    _simd
	a, b uint64
}

// LoadInt64s loads a slice of int64 into an Int64s vector.
func LoadInt64s(s []int64) Int64s {
	var a, b uint64
	a = uint64(s[0])
	b = uint64(s[1])
	return Int64s{a: a, b: b}
}

// LoadInt64sPart loads a partial slice of int64 into an Int64s vector.
func LoadInt64sPart(s []int64) (Int64s, int) {
	var a, b uint64
	if len(s) > 0 {
		a = uint64(s[0])
	}
	if len(s) > 1 {
		b = uint64(s[1])
	}
	return Int64s{a: a, b: b}, len(s)
}

func (x Int64s) get(i int) int64 {
	if i == 0 {
		return int64(x.a)
	}
	return int64(x.b)
}

func (x *Int64s) set(i int, v int64) {
	if i == 0 {
		x.a = uint64(v)
	} else {
		x.b = uint64(v)
	}
}

// Add returns the element-wise sum of x and y.
func (x Int64s) Add(y Int64s) Int64s {
	return Int64s{a: x.a + y.a, b: x.b + y.b}
}

// And returns the bitwise AND of x and y.
func (x Int64s) And(y Int64s) Int64s {
	return Int64s{a: x.a & y.a, b: x.b & y.b}
}

// AndNot returns the bitwise AND NOT of x and y.
func (x Int64s) AndNot(y Int64s) Int64s {
	return Int64s{a: x.a &^ y.a, b: x.b &^ y.b}
}

// Equal returns a mask indicating where x and y are equal.
func (x Int64s) Equal(y Int64s) Mask64s {
	var res Mask64s
	if x.a == y.a {
		res.a = ^uint64(0)
	}
	if x.b == y.b {
		res.b = ^uint64(0)
	}
	return res
}

// Greater returns a mask indicating where x is greater than y.
func (x Int64s) Greater(y Int64s) Mask64s {
	var res Mask64s
	if int64(x.a) > int64(y.a) {
		res.a = ^uint64(0)
	}
	if int64(x.b) > int64(y.b) {
		res.b = ^uint64(0)
	}
	return res
}

// GreaterEqual returns a mask indicating where x is greater than or equal to y.
func (x Int64s) GreaterEqual(y Int64s) Mask64s {
	var res Mask64s
	if int64(x.a) >= int64(y.a) {
		res.a = ^uint64(0)
	}
	if int64(x.b) >= int64(y.b) {
		res.b = ^uint64(0)
	}
	return res
}

// Less returns a mask indicating where x is less than y.
func (x Int64s) Less(y Int64s) Mask64s {
	var res Mask64s
	if int64(x.a) < int64(y.a) {
		res.a = ^uint64(0)
	}
	if int64(x.b) < int64(y.b) {
		res.b = ^uint64(0)
	}
	return res
}

// LessEqual returns a mask indicating where x is less than or equal to y.
func (x Int64s) LessEqual(y Int64s) Mask64s {
	var res Mask64s
	if int64(x.a) <= int64(y.a) {
		res.a = ^uint64(0)
	}
	if int64(x.b) <= int64(y.b) {
		res.b = ^uint64(0)
	}
	return res
}

// NotEqual returns a mask indicating where x and y are not equal.
func (x Int64s) NotEqual(y Int64s) Mask64s {
	var res Mask64s
	if x.a != y.a {
		res.a = ^uint64(0)
	}
	if x.b != y.b {
		res.b = ^uint64(0)
	}
	return res
}

// Len returns the number of elements in the vector.
func (x Int64s) Len() int {
	return 2
}

// Masked returns a new vector with elements from x where mask is true, and zero elsewhere.
func (x Int64s) Masked(mask Mask64s) Int64s {
	return Int64s{a: x.a & mask.a, b: x.b & mask.b}
}

// IfElse returns a new vector with elements from x where mask is true, and y where mask is false.
func (x Int64s) IfElse(mask Mask64s, y Int64s) Int64s {
	return Int64s{
		a: (x.a & mask.a) | (y.a &^ mask.a),
		b: (x.b & mask.b) | (y.b &^ mask.b),
	}
}

// Neg returns the element-wise negation of x.
func (x Int64s) Neg() Int64s {
	return Int64s{a: uint64(-int64(x.a)), b: uint64(-int64(x.b))}
}

// Not returns the bitwise NOT of x.
func (x Int64s) Not() Int64s {
	return Int64s{a: ^x.a, b: ^x.b}
}

// Or returns the bitwise OR of x and y.
func (x Int64s) Or(y Int64s) Int64s {
	return Int64s{a: x.a | y.a, b: x.b | y.b}
}

// ShiftAllLeft shifts all elements left by y bits.
func (x Int64s) ShiftAllLeft(y uint8) Int64s {
	return Int64s{a: x.a << y, b: x.b << y}
}

// RotateAllLeft rotates all elements left by dist bits.
func (x Int64s) RotateAllLeft(dist uint64) Int64s {
	d := dist & 63
	return Int64s{
		a: (x.a << d) | (x.a >> ((64 - d) & 63)),
		b: (x.b << d) | (x.b >> ((64 - d) & 63)),
	}
}

// RotateAllRight rotates all elements right by dist bits.
func (x Int64s) RotateAllRight(dist uint64) Int64s {
	d := dist & 63
	return Int64s{
		a: (x.a >> d) | (x.a << ((64 - d) & 63)),
		b: (x.b >> d) | (x.b << ((64 - d) & 63)),
	}
}

// Store stores the vector elements into the slice s.
func (x Int64s) Store(s []int64) {
	if len(s) > 0 {
		s[0] = int64(x.a)
	}
	if len(s) > 1 {
		s[1] = int64(x.b)
	}
}

// StorePart stores a partial vector into the slice s.
func (x Int64s) StorePart(s []int64) {
	x.Store(s)
}

// String returns a string representation of the vector.
func (x Int64s) String() string {
	return fmt.Sprint([2]int64{int64(x.a), int64(x.b)})
}

// Sub returns the element-wise difference of x and y.
func (x Int64s) Sub(y Int64s) Int64s {
	return Int64s{a: x.a - y.a, b: x.b - y.b}
}

// ToMask returns a mask representation of the vector.
func (x Int64s) ToMask() Mask64s {
	var res Mask64s
	if x.a != 0 {
		res.a = ^uint64(0)
	}
	if x.b != 0 {
		res.b = ^uint64(0)
	}
	return res
}

// Xor returns the bitwise XOR of x and y.
func (x Int64s) Xor(y Int64s) Int64s {
	return Int64s{a: x.a ^ y.a, b: x.b ^ y.b}
}

// ConvertToUint64 converts the vector elements to uint64.
func (x Int64s) ConvertToUint64() Uint64s {
	return Uint64s{a: x.a, b: x.b}
}

// ToBits reinterprets the vector bits as a Uint64s vector.
func (x Int64s) ToBits() Uint64s {
	return Uint64s{a: x.a, b: x.b}
}

// Uint8s represents a 128-bit vector of 16 uint8 elements.
type Uint8s struct {
	_    _simd
	a, b uint64
}

// LoadUint8s loads a slice of uint8 into an Uint8s vector.
func LoadUint8s(s []uint8) Uint8s {
	var a, b uint64
	for i := 0; i < 16; i++ {
		val := uint64(s[i])
		if i < 8 {
			a |= val << (8 * i)
		} else {
			b |= val << (8 * (i - 8))
		}
	}
	return Uint8s{a: a, b: b}
}

// LoadUint8sPart loads a partial slice of uint8 into an Uint8s vector.
func LoadUint8sPart(s []uint8) (Uint8s, int) {
	var a, b uint64
	n := len(s)
	if n > 16 {
		n = 16
	}
	for i := 0; i < n; i++ {
		val := uint64(s[i])
		if i < 8 {
			a |= val << (8 * i)
		} else {
			b |= val << (8 * (i - 8))
		}
	}
	return Uint8s{a: a, b: b}, n
}

func (x Uint8s) get(i int) uint8 {
	if i < 8 {
		return uint8(x.a >> (8 * i))
	}
	return uint8(x.b >> (8 * (i - 8)))
}

func (x *Uint8s) set(i int, v uint8) {
	val := uint64(v)
	if i < 8 {
		mask := uint64(0xff) << (8 * i)
		x.a = (x.a &^ mask) | (val << (8 * i))
	} else {
		mask := uint64(0xff) << (8 * (i - 8))
		x.b = (x.b &^ mask) | (val << (8 * (i - 8)))
	}
}

// Add returns the element-wise sum of x and y.
func (x Uint8s) Add(y Uint8s) Uint8s {
	var res Uint8s
	for i := 0; i < 16; i++ {
		res.set(i, x.get(i)+y.get(i))
	}
	return res
}

// AddSaturated returns the element-wise saturated sum of x and y.
func (x Uint8s) AddSaturated(y Uint8s) Uint8s {
	var res Uint8s
	for i := 0; i < 16; i++ {
		sum := int(x.get(i)) + int(y.get(i))
		if sum > math.MaxUint8 {
			res.set(i, math.MaxUint8)
		} else {
			res.set(i, uint8(sum))
		}
	}
	return res
}

// And returns the bitwise AND of x and y.
func (x Uint8s) And(y Uint8s) Uint8s {
	return Uint8s{a: x.a & y.a, b: x.b & y.b}
}

// AndNot returns the bitwise AND NOT of x and y.
func (x Uint8s) AndNot(y Uint8s) Uint8s {
	return Uint8s{a: x.a &^ y.a, b: x.b &^ y.b}
}

// Average returns the element-wise average of x and y.
func (x Uint8s) Average(y Uint8s) Uint8s {
	var res Uint8s
	for i := 0; i < 16; i++ {
		res.set(i, uint8((int(x.get(i))+int(y.get(i))+1)>>1))
	}
	return res
}

// Equal returns a mask indicating where x and y are equal.
func (x Uint8s) Equal(y Uint8s) Mask8s {
	var res Mask8s
	for i := 0; i < 16; i++ {
		if x.get(i) == y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// NotEqual returns a mask indicating where x and y are not equal.
func (x Uint8s) NotEqual(y Uint8s) Mask8s {
	var res Mask8s
	for i := 0; i < 16; i++ {
		if x.get(i) != y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// Len returns the number of elements in the vector.
func (x Uint8s) Len() int {
	return 16
}

// Masked returns a new vector with elements from x where mask is true, and zero elsewhere.
func (x Uint8s) Masked(mask Mask8s) Uint8s {
	return Uint8s{a: x.a & mask.a, b: x.b & mask.b}
}

// Max returns the element-wise maximum of x and y.
func (x Uint8s) Max(y Uint8s) Uint8s {
	var res Uint8s
	for i := 0; i < 16; i++ {
		vx := x.get(i)
		vy := y.get(i)
		if vx > vy {
			res.set(i, vx)
		} else {
			res.set(i, vy)
		}
	}
	return res
}

// IfElse returns a new vector with elements from x where mask is true, and y where mask is false.
func (x Uint8s) IfElse(mask Mask8s, y Uint8s) Uint8s {
	return Uint8s{
		a: (x.a & mask.a) | (y.a &^ mask.a),
		b: (x.b & mask.b) | (y.b &^ mask.b),
	}
}

// Min returns the element-wise minimum of x and y.
func (x Uint8s) Min(y Uint8s) Uint8s {
	var res Uint8s
	for i := 0; i < 16; i++ {
		vx := x.get(i)
		vy := y.get(i)
		if vx < vy {
			res.set(i, vx)
		} else {
			res.set(i, vy)
		}
	}
	return res
}

// Mul returns the element-wise product of x and y.
func (x Uint8s) Mul(y Uint8s) Uint8s {
	var res Uint8s
	for i := 0; i < 16; i++ {
		res.set(i, x.get(i)*y.get(i))
	}
	return res
}

// Not returns the bitwise NOT of x.
func (x Uint8s) Not() Uint8s {
	return Uint8s{a: ^x.a, b: ^x.b}
}

// Or returns the bitwise OR of x and y.
func (x Uint8s) Or(y Uint8s) Uint8s {
	return Uint8s{a: x.a | y.a, b: x.b | y.b}
}

// Store stores the vector elements into the slice s.
func (x Uint8s) Store(s []uint8) {
	for i := 0; i < 16 && i < len(s); i++ {
		s[i] = x.get(i)
	}
}

// StorePart stores a partial vector into the slice s.
func (x Uint8s) StorePart(s []uint8) {
	x.Store(s)
}

// String returns a string representation of the vector.
func (x Uint8s) String() string {
	var parts [16]uint8
	for i := 0; i < 16; i++ {
		parts[i] = x.get(i)
	}
	return fmt.Sprint(parts)
}

// Sub returns the element-wise difference of x and y.
func (x Uint8s) Sub(y Uint8s) Uint8s {
	var res Uint8s
	for i := 0; i < 16; i++ {
		res.set(i, x.get(i)-y.get(i))
	}
	return res
}

// SubSaturated returns the element-wise saturated difference of x and y.
func (x Uint8s) SubSaturated(y Uint8s) Uint8s {
	var res Uint8s
	for i := 0; i < 16; i++ {
		vx := x.get(i)
		vy := y.get(i)
		if vx < vy {
			res.set(i, 0)
		} else {
			res.set(i, vx-vy)
		}
	}
	return res
}

// Xor returns the bitwise XOR of x and y.
func (x Uint8s) Xor(y Uint8s) Uint8s {
	return Uint8s{a: x.a ^ y.a, b: x.b ^ y.b}
}

// BitsToInt8 reinterprets the vector bits as an Int8s vector.
func (x Uint8s) BitsToInt8() Int8s {
	return Int8s{a: x.a, b: x.b}
}

// ConvertToInt8 converts the vector elements to int8.
func (x Uint8s) ConvertToInt8() Int8s {
	return Int8s{a: x.a, b: x.b}
}

// ReshapeToUint16s reinterprets the vector bits as a Uint16s vector.
func (x Uint8s) ReshapeToUint16s() Uint16s {
	return Uint16s{a: x.a, b: x.b}
}

// ReshapeToUint32s reinterprets the vector bits as a Uint32s vector.
func (x Uint8s) ReshapeToUint32s() Uint32s {
	return Uint32s{a: x.a, b: x.b}
}

// ReshapeToUint64s reinterprets the vector bits as a Uint64s vector.
func (x Uint8s) ReshapeToUint64s() Uint64s {
	return Uint64s{a: x.a, b: x.b}
}

// Uint16s represents a 128-bit vector of 8 uint16 elements.
type Uint16s struct {
	_    _simd
	a, b uint64
}

// LoadUint16s loads a slice of uint16 into an Uint16s vector.
func LoadUint16s(s []uint16) Uint16s {
	var a, b uint64
	for i := 0; i < 8; i++ {
		val := uint64(s[i])
		if i < 4 {
			a |= val << (16 * i)
		} else {
			b |= val << (16 * (i - 4))
		}
	}
	return Uint16s{a: a, b: b}
}

// LoadUint16sPart loads a partial slice of uint16 into an Uint16s vector.
func LoadUint16sPart(s []uint16) (Uint16s, int) {
	var a, b uint64
	n := len(s)
	if n > 8 {
		n = 8
	}
	for i := 0; i < n; i++ {
		val := uint64(s[i])
		if i < 4 {
			a |= val << (16 * i)
		} else {
			b |= val << (16 * (i - 4))
		}
	}
	return Uint16s{a: a, b: b}, n
}

func (x Uint16s) get(i int) uint16 {
	if i < 4 {
		return uint16(x.a >> (16 * i))
	}
	return uint16(x.b >> (16 * (i - 4)))
}

func (x *Uint16s) set(i int, v uint16) {
	val := uint64(v)
	if i < 4 {
		mask := uint64(0xffff) << (16 * i)
		x.a = (x.a &^ mask) | (val << (16 * i))
	} else {
		mask := uint64(0xffff) << (16 * (i - 4))
		x.b = (x.b &^ mask) | (val << (16 * (i - 4)))
	}
}

// Add returns the element-wise sum of x and y.
func (x Uint16s) Add(y Uint16s) Uint16s {
	var res Uint16s
	for i := 0; i < 8; i++ {
		res.set(i, x.get(i)+y.get(i))
	}
	return res
}

// AddSaturated returns the element-wise saturated sum of x and y.
func (x Uint16s) AddSaturated(y Uint16s) Uint16s {
	var res Uint16s
	for i := 0; i < 8; i++ {
		sum := int(x.get(i)) + int(y.get(i))
		if sum > math.MaxUint16 {
			res.set(i, math.MaxUint16)
		} else {
			res.set(i, uint16(sum))
		}
	}
	return res
}

// And returns the bitwise AND of x and y.
func (x Uint16s) And(y Uint16s) Uint16s {
	return Uint16s{a: x.a & y.a, b: x.b & y.b}
}

// AndNot returns the bitwise AND NOT of x and y.
func (x Uint16s) AndNot(y Uint16s) Uint16s {
	return Uint16s{a: x.a &^ y.a, b: x.b &^ y.b}
}

// Average returns the element-wise average of x and y.
func (x Uint16s) Average(y Uint16s) Uint16s {
	var res Uint16s
	for i := 0; i < 8; i++ {
		res.set(i, uint16((int(x.get(i))+int(y.get(i))+1)>>1))
	}
	return res
}

// Equal returns a mask indicating where x and y are equal.
func (x Uint16s) Equal(y Uint16s) Mask16s {
	var res Mask16s
	for i := 0; i < 8; i++ {
		if x.get(i) == y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// Greater returns a mask indicating where x is greater than y.
func (x Uint16s) Greater(y Uint16s) Mask16s {
	var res Mask16s
	for i := 0; i < 8; i++ {
		if x.get(i) > y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// GreaterEqual returns a mask indicating where x is greater than or equal to y.
func (x Uint16s) GreaterEqual(y Uint16s) Mask16s {
	var res Mask16s
	for i := 0; i < 8; i++ {
		if x.get(i) >= y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// Less returns a mask indicating where x is less than y.
func (x Uint16s) Less(y Uint16s) Mask16s {
	var res Mask16s
	for i := 0; i < 8; i++ {
		if x.get(i) < y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// LessEqual returns a mask indicating where x is less than or equal to y.
func (x Uint16s) LessEqual(y Uint16s) Mask16s {
	var res Mask16s
	for i := 0; i < 8; i++ {
		if x.get(i) <= y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// NotEqual returns a mask indicating where x and y are not equal.
func (x Uint16s) NotEqual(y Uint16s) Mask16s {
	var res Mask16s
	for i := 0; i < 8; i++ {
		if x.get(i) != y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// Len returns the number of elements in the vector.
func (x Uint16s) Len() int {
	return 8
}

// Masked returns a new vector with elements from x where mask is true, and zero elsewhere.
func (x Uint16s) Masked(mask Mask16s) Uint16s {
	return Uint16s{a: x.a & mask.a, b: x.b & mask.b}
}

// Max returns the element-wise maximum of x and y.
func (x Uint16s) Max(y Uint16s) Uint16s {
	var res Uint16s
	for i := 0; i < 8; i++ {
		vx := x.get(i)
		vy := y.get(i)
		if vx > vy {
			res.set(i, vx)
		} else {
			res.set(i, vy)
		}
	}
	return res
}

// IfElse returns a new vector with elements from x where mask is true, and y where mask is false.
func (x Uint16s) IfElse(mask Mask16s, y Uint16s) Uint16s {
	return Uint16s{
		a: (x.a & mask.a) | (y.a &^ mask.a),
		b: (x.b & mask.b) | (y.b &^ mask.b),
	}
}

// Min returns the element-wise minimum of x and y.
func (x Uint16s) Min(y Uint16s) Uint16s {
	var res Uint16s
	for i := 0; i < 8; i++ {
		vx := x.get(i)
		vy := y.get(i)
		if vx < vy {
			res.set(i, vx)
		} else {
			res.set(i, vy)
		}
	}
	return res
}

// Mul returns the element-wise product of x and y.
func (x Uint16s) Mul(y Uint16s) Uint16s {
	var res Uint16s
	for i := 0; i < 8; i++ {
		res.set(i, x.get(i)*y.get(i))
	}
	return res
}

// Not returns the bitwise NOT of x.
func (x Uint16s) Not() Uint16s {
	return Uint16s{a: ^x.a, b: ^x.b}
}

// Or returns the bitwise OR of x and y.
func (x Uint16s) Or(y Uint16s) Uint16s {
	return Uint16s{a: x.a | y.a, b: x.b | y.b}
}

// ShiftAllLeft shifts all elements left by y bits.
func (x Uint16s) ShiftAllLeft(y uint8) Uint16s {
	var res Uint16s
	for i := 0; i < 8; i++ {
		res.set(i, x.get(i)<<y)
	}
	return res
}

// ShiftAllRight shifts all elements right by y bits.
func (x Uint16s) ShiftAllRight(y uint8) Uint16s {
	var res Uint16s
	for i := 0; i < 8; i++ {
		res.set(i, x.get(i)>>y)
	}
	return res
}

// RotateAllLeft rotates all elements left by dist bits.
func (x Uint16s) RotateAllLeft(dist uint64) Uint16s {
	var res Uint16s
	d := dist & 15
	for i := 0; i < 8; i++ {
		u := x.get(i)
		r := (u << d) | (u >> ((16 - d) & 15))
		res.set(i, r)
	}
	return res
}

// RotateAllRight rotates all elements right by dist bits.
func (x Uint16s) RotateAllRight(dist uint64) Uint16s {
	var res Uint16s
	d := dist & 15
	for i := 0; i < 8; i++ {
		u := x.get(i)
		r := (u >> d) | (u << ((16 - d) & 15))
		res.set(i, r)
	}
	return res
}

// Store stores the vector elements into the slice s.
func (x Uint16s) Store(s []uint16) {
	for i := 0; i < 8 && i < len(s); i++ {
		s[i] = x.get(i)
	}
}

// StorePart stores a partial vector into the slice s.
func (x Uint16s) StorePart(s []uint16) {
	x.Store(s)
}

// String returns a string representation of the vector.
func (x Uint16s) String() string {
	var parts [8]uint16
	for i := 0; i < 8; i++ {
		parts[i] = x.get(i)
	}
	return fmt.Sprint(parts)
}

// Sub returns the element-wise difference of x and y.
func (x Uint16s) Sub(y Uint16s) Uint16s {
	var res Uint16s
	for i := 0; i < 8; i++ {
		res.set(i, x.get(i)-y.get(i))
	}
	return res
}

// SubSaturated returns the element-wise saturated difference of x and y.
func (x Uint16s) SubSaturated(y Uint16s) Uint16s {
	var res Uint16s
	for i := 0; i < 8; i++ {
		vx := x.get(i)
		vy := y.get(i)
		if vx < vy {
			res.set(i, 0)
		} else {
			res.set(i, vx-vy)
		}
	}
	return res
}

// Xor returns the bitwise XOR of x and y.
func (x Uint16s) Xor(y Uint16s) Uint16s {
	return Uint16s{a: x.a ^ y.a, b: x.b ^ y.b}
}

// BitsToInt16 reinterprets the vector bits as an Int16s vector.
func (x Uint16s) BitsToInt16() Int16s {
	return Int16s{a: x.a, b: x.b}
}

// ConvertToInt16 converts the vector elements to int16.
func (x Uint16s) ConvertToInt16() Int16s {
	return Int16s{a: x.a, b: x.b}
}

// ReshapeToUint32s reinterprets the vector bits as a Uint32s vector.
func (x Uint16s) ReshapeToUint32s() Uint32s {
	return Uint32s{a: x.a, b: x.b}
}

// ReshapeToUint64s reinterprets the vector bits as a Uint64s vector.
func (x Uint16s) ReshapeToUint64s() Uint64s {
	return Uint64s{a: x.a, b: x.b}
}

// ReshapeToUint8s reinterprets the vector bits as a Uint8s vector.
func (x Uint16s) ReshapeToUint8s() Uint8s {
	return Uint8s{a: x.a, b: x.b}
}

// Uint32s represents a 128-bit vector of 4 uint32 elements.
type Uint32s struct {
	_    _simd
	a, b uint64
}

// LoadUint32s loads a slice of uint32 into an Uint32s vector.
func LoadUint32s(s []uint32) Uint32s {
	var a, b uint64
	for i := 0; i < 4; i++ {
		val := uint64(s[i])
		if i < 2 {
			a |= val << (32 * i)
		} else {
			b |= val << (32 * (i - 2))
		}
	}
	return Uint32s{a: a, b: b}
}

// LoadUint32sPart loads a partial slice of uint32 into an Uint32s vector.
func LoadUint32sPart(s []uint32) (Uint32s, int) {
	var a, b uint64
	n := len(s)
	if n > 4 {
		n = 4
	}
	for i := 0; i < n; i++ {
		val := uint64(s[i])
		if i < 2 {
			a |= val << (32 * i)
		} else {
			b |= val << (32 * (i - 2))
		}
	}
	return Uint32s{a: a, b: b}, n
}

func (x Uint32s) get(i int) uint32 {
	if i < 2 {
		return uint32(x.a >> (32 * i))
	}
	return uint32(x.b >> (32 * (i - 2)))
}

func (x *Uint32s) set(i int, v uint32) {
	val := uint64(v)
	if i < 2 {
		mask := uint64(0xffffffff) << (32 * i)
		x.a = (x.a &^ mask) | (val << (32 * i))
	} else {
		mask := uint64(0xffffffff) << (32 * (i - 2))
		x.b = (x.b &^ mask) | (val << (32 * (i - 2)))
	}
}

// Add returns the element-wise sum of x and y.
func (x Uint32s) Add(y Uint32s) Uint32s {
	var res Uint32s
	for i := 0; i < 4; i++ {
		res.set(i, x.get(i)+y.get(i))
	}
	return res
}

// And returns the bitwise AND of x and y.
func (x Uint32s) And(y Uint32s) Uint32s {
	return Uint32s{a: x.a & y.a, b: x.b & y.b}
}

// AndNot returns the bitwise AND NOT of x and y.
func (x Uint32s) AndNot(y Uint32s) Uint32s {
	return Uint32s{a: x.a &^ y.a, b: x.b &^ y.b}
}

// Equal returns a mask indicating where x and y are equal.
func (x Uint32s) Equal(y Uint32s) Mask32s {
	var res Mask32s
	for i := 0; i < 4; i++ {
		if x.get(i) == y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// Greater returns a mask indicating where x is greater than y.
func (x Uint32s) Greater(y Uint32s) Mask32s {
	var res Mask32s
	for i := 0; i < 4; i++ {
		if x.get(i) > y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// GreaterEqual returns a mask indicating where x is greater than or equal to y.
func (x Uint32s) GreaterEqual(y Uint32s) Mask32s {
	var res Mask32s
	for i := 0; i < 4; i++ {
		if x.get(i) >= y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// Less returns a mask indicating where x is less than y.
func (x Uint32s) Less(y Uint32s) Mask32s {
	var res Mask32s
	for i := 0; i < 4; i++ {
		if x.get(i) < y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// LessEqual returns a mask indicating where x is less than or equal to y.
func (x Uint32s) LessEqual(y Uint32s) Mask32s {
	var res Mask32s
	for i := 0; i < 4; i++ {
		if x.get(i) <= y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// NotEqual returns a mask indicating where x and y are not equal.
func (x Uint32s) NotEqual(y Uint32s) Mask32s {
	var res Mask32s
	for i := 0; i < 4; i++ {
		if x.get(i) != y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// Len returns the number of elements in the vector.
func (x Uint32s) Len() int {
	return 4
}

// Masked returns a new vector with elements from x where mask is true, and zero elsewhere.
func (x Uint32s) Masked(mask Mask32s) Uint32s {
	return Uint32s{a: x.a & mask.a, b: x.b & mask.b}
}

// Max returns the element-wise maximum of x and y.
func (x Uint32s) Max(y Uint32s) Uint32s {
	var res Uint32s
	for i := 0; i < 4; i++ {
		vx := x.get(i)
		vy := y.get(i)
		if vx > vy {
			res.set(i, vx)
		} else {
			res.set(i, vy)
		}
	}
	return res
}

// IfElse returns a new vector with elements from x where mask is true, and y where mask is false.
func (x Uint32s) IfElse(mask Mask32s, y Uint32s) Uint32s {
	return Uint32s{
		a: (x.a & mask.a) | (y.a &^ mask.a),
		b: (x.b & mask.b) | (y.b &^ mask.b),
	}
}

// Min returns the element-wise minimum of x and y.
func (x Uint32s) Min(y Uint32s) Uint32s {
	var res Uint32s
	for i := 0; i < 4; i++ {
		vx := x.get(i)
		vy := y.get(i)
		if vx < vy {
			res.set(i, vx)
		} else {
			res.set(i, vy)
		}
	}
	return res
}

// Mul returns the element-wise product of x and y.
func (x Uint32s) Mul(y Uint32s) Uint32s {
	var res Uint32s
	for i := 0; i < 4; i++ {
		res.set(i, x.get(i)*y.get(i))
	}
	return res
}

// Not returns the bitwise NOT of x.
func (x Uint32s) Not() Uint32s {
	return Uint32s{a: ^x.a, b: ^x.b}
}

// Or returns the bitwise OR of x and y.
func (x Uint32s) Or(y Uint32s) Uint32s {
	return Uint32s{a: x.a | y.a, b: x.b | y.b}
}

// ShiftAllLeft shifts all elements left by y bits.
func (x Uint32s) ShiftAllLeft(y uint8) Uint32s {
	var res Uint32s
	for i := 0; i < 4; i++ {
		res.set(i, x.get(i)<<y)
	}
	return res
}

// ShiftAllRight shifts all elements right by y bits.
func (x Uint32s) ShiftAllRight(y uint8) Uint32s {
	var res Uint32s
	for i := 0; i < 4; i++ {
		res.set(i, x.get(i)>>y)
	}
	return res
}

// RotateAllLeft rotates all elements left by dist bits.
func (x Uint32s) RotateAllLeft(dist uint64) Uint32s {
	var res Uint32s
	d := dist & 31
	for i := 0; i < 4; i++ {
		u := x.get(i)
		r := (u << d) | (u >> ((32 - d) & 31))
		res.set(i, r)
	}
	return res
}

// RotateAllRight rotates all elements right by dist bits.
func (x Uint32s) RotateAllRight(dist uint64) Uint32s {
	var res Uint32s
	d := dist & 31
	for i := 0; i < 4; i++ {
		u := x.get(i)
		r := (u >> d) | (u << ((32 - d) & 31))
		res.set(i, r)
	}
	return res
}

// Store stores the vector elements into the slice s.
func (x Uint32s) Store(s []uint32) {
	for i := 0; i < 4 && i < len(s); i++ {
		s[i] = x.get(i)
	}
}

// StorePart stores a partial vector into the slice s.
func (x Uint32s) StorePart(s []uint32) {
	x.Store(s)
}

// String returns a string representation of the vector.
func (x Uint32s) String() string {
	var parts [4]uint32
	for i := 0; i < 4; i++ {
		parts[i] = x.get(i)
	}
	return fmt.Sprint(parts)
}

// Sub returns the element-wise difference of x and y.
func (x Uint32s) Sub(y Uint32s) Uint32s {
	var res Uint32s
	for i := 0; i < 4; i++ {
		res.set(i, x.get(i)-y.get(i))
	}
	return res
}

// Xor returns the bitwise XOR of x and y.
func (x Uint32s) Xor(y Uint32s) Uint32s {
	return Uint32s{a: x.a ^ y.a, b: x.b ^ y.b}
}

// BitsToFloat32 reinterprets the vector bits as a Float32s vector.
func (x Uint32s) BitsToFloat32() Float32s {
	return Float32s{a: x.a, b: x.b}
}

// BitsToInt32 reinterprets the vector bits as an Int32s vector.
func (x Uint32s) BitsToInt32() Int32s {
	return Int32s{a: x.a, b: x.b}
}

// ConvertToInt32 converts the vector elements to int32.
func (x Uint32s) ConvertToInt32() Int32s {
	return Int32s{a: x.a, b: x.b}
}

// ReshapeToUint16s reinterprets the vector bits as a Uint16s vector.
func (x Uint32s) ReshapeToUint16s() Uint16s {
	return Uint16s{a: x.a, b: x.b}
}

// ReshapeToUint64s reinterprets the vector bits as a Uint64s vector.
func (x Uint32s) ReshapeToUint64s() Uint64s {
	return Uint64s{a: x.a, b: x.b}
}

// ReshapeToUint8s reinterprets the vector bits as a Uint8s vector.
func (x Uint32s) ReshapeToUint8s() Uint8s {
	return Uint8s{a: x.a, b: x.b}
}

// Uint64s represents a 128-bit vector of 2 uint64 elements.
type Uint64s struct {
	_    _simd
	a, b uint64
}

// LoadUint64s loads a slice of uint64 into an Uint64s vector.
func LoadUint64s(s []uint64) Uint64s {
	var a, b uint64
	a = s[0]
	b = s[1]
	return Uint64s{a: a, b: b}
}

// LoadUint64sPart loads a partial slice of uint64 into an Uint64s vector.
func LoadUint64sPart(s []uint64) (Uint64s, int) {
	n := len(s)
	var a, b uint64
	if n > 0 {
		a = s[0]
	}
	if n > 1 {
		b = s[1]
	}
	return Uint64s{a: a, b: b}, n
}

func (x Uint64s) get(i int) uint64 {
	if i == 0 {
		return x.a
	}
	return x.b
}

func (x *Uint64s) set(i int, v uint64) {
	if i == 0 {
		x.a = v
	} else {
		x.b = v
	}
}

// Add returns the element-wise sum of x and y.
func (x Uint64s) Add(y Uint64s) Uint64s {
	return Uint64s{a: x.a + y.a, b: x.b + y.b}
}

// And returns the bitwise AND of x and y.
func (x Uint64s) And(y Uint64s) Uint64s {
	return Uint64s{a: x.a & y.a, b: x.b & y.b}
}

// AndNot returns the bitwise AND NOT of x and y.
func (x Uint64s) AndNot(y Uint64s) Uint64s {
	return Uint64s{a: x.a &^ y.a, b: x.b &^ y.b}
}

// Equal returns a mask indicating where x and y are equal.
func (x Uint64s) Equal(y Uint64s) Mask64s {
	var res Mask64s
	if x.a == y.a {
		res.a = ^uint64(0)
	}
	if x.b == y.b {
		res.b = ^uint64(0)
	}
	return res
}

// Greater returns a mask indicating where x is greater than y.
func (x Uint64s) Greater(y Uint64s) Mask64s {
	var res Mask64s
	for i := 0; i < 2; i++ {
		if x.get(i) > y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// GreaterEqual returns a mask indicating where x is greater than or equal to y.
func (x Uint64s) GreaterEqual(y Uint64s) Mask64s {
	var res Mask64s
	for i := 0; i < 2; i++ {
		if x.get(i) >= y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// Less returns a mask indicating where x is less than y.
func (x Uint64s) Less(y Uint64s) Mask64s {
	var res Mask64s
	for i := 0; i < 2; i++ {
		if x.get(i) < y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// LessEqual returns a mask indicating where x is less than or equal to y.
func (x Uint64s) LessEqual(y Uint64s) Mask64s {
	var res Mask64s
	for i := 0; i < 2; i++ {
		if x.get(i) <= y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// NotEqual returns a mask indicating where x and y are not equal.
func (x Uint64s) NotEqual(y Uint64s) Mask64s {
	var res Mask64s
	if x.a != y.a {
		res.a = ^uint64(0)
	}
	if x.b != y.b {
		res.b = ^uint64(0)
	}
	return res
}

// Len returns the number of elements in the vector.
func (x Uint64s) Len() int {
	return 2
}

// Masked returns a new vector with elements from x where mask is true, and zero elsewhere.
func (x Uint64s) Masked(mask Mask64s) Uint64s {
	return Uint64s{a: x.a & mask.a, b: x.b & mask.b}
}

// IfElse returns a new vector with elements from x where mask is true, and y where mask is false.
func (x Uint64s) IfElse(mask Mask64s, y Uint64s) Uint64s {
	return Uint64s{
		a: (x.a & mask.a) | (y.a &^ mask.a),
		b: (x.b & mask.b) | (y.b &^ mask.b),
	}
}

// Not returns the bitwise NOT of x.
func (x Uint64s) Not() Uint64s {
	return Uint64s{a: ^x.a, b: ^x.b}
}

// Or returns the bitwise OR of x and y.
func (x Uint64s) Or(y Uint64s) Uint64s {
	return Uint64s{a: x.a | y.a, b: x.b | y.b}
}

// ShiftAllLeft shifts all elements left by y bits.
func (x Uint64s) ShiftAllLeft(y uint8) Uint64s {
	return Uint64s{a: x.a << y, b: x.b << y}
}

// ShiftAllRight shifts all elements right by y bits.
func (x Uint64s) ShiftAllRight(y uint8) Uint64s {
	return Uint64s{a: x.a >> y, b: x.b >> y}
}

// RotateAllLeft rotates all elements left by dist bits.
func (x Uint64s) RotateAllLeft(dist uint64) Uint64s {
	d := dist & 63
	return Uint64s{
		a: (x.a << d) | (x.a >> ((64 - d) & 63)),
		b: (x.b << d) | (x.b >> ((64 - d) & 63)),
	}
}

// RotateAllRight rotates all elements right by dist bits.
func (x Uint64s) RotateAllRight(dist uint64) Uint64s {
	d := dist & 63
	return Uint64s{
		a: (x.a >> d) | (x.a << ((64 - d) & 63)),
		b: (x.b >> d) | (x.b << ((64 - d) & 63)),
	}
}

// Store stores the vector elements into the slice s.
func (x Uint64s) Store(s []uint64) {
	if len(s) > 0 {
		s[0] = x.a
	}
	if len(s) > 1 {
		s[1] = x.b
	}
}

// StorePart stores a partial vector into the slice s.
func (x Uint64s) StorePart(s []uint64) {
	x.Store(s)
}

// String returns a string representation of the vector.
func (x Uint64s) String() string {
	return fmt.Sprint([2]uint64{x.a, x.b})
}

// Sub returns the element-wise difference of x and y.
func (x Uint64s) Sub(y Uint64s) Uint64s {
	return Uint64s{a: x.a - y.a, b: x.b - y.b}
}

// Xor returns the bitwise XOR of x and y.
func (x Uint64s) Xor(y Uint64s) Uint64s {
	return Uint64s{a: x.a ^ y.a, b: x.b ^ y.b}
}

// BitsToFloat64 reinterprets the vector bits as a Float64s vector.
func (x Uint64s) BitsToFloat64() Float64s {
	return Float64s{a: x.a, b: x.b}
}

// BitsToInt64 reinterprets the vector bits as an Int64s vector.
func (x Uint64s) BitsToInt64() Int64s {
	return Int64s{a: x.a, b: x.b}
}

// ConvertToInt64 converts the vector elements to int64.
func (x Uint64s) ConvertToInt64() Int64s {
	return Int64s{a: x.a, b: x.b}
}

// ReshapeToUint16s reinterprets the vector bits as a Uint16s vector.
func (x Uint64s) ReshapeToUint16s() Uint16s {
	return Uint16s{a: x.a, b: x.b}
}

// ReshapeToUint32s reinterprets the vector bits as a Uint32s vector.
func (x Uint64s) ReshapeToUint32s() Uint32s {
	return Uint32s{a: x.a, b: x.b}
}

// ReshapeToUint8s reinterprets the vector bits as a Uint8s vector.
func (x Uint64s) ReshapeToUint8s() Uint8s {
	return Uint8s{a: x.a, b: x.b}
}

// Float32s represents a 128-bit vector of 4 float32 elements.
type Float32s struct {
	_    _simd
	a, b uint64
}

// LoadFloat32s loads a slice of float32 into an Float32s vector.
func LoadFloat32s(s []float32) Float32s {
	var a, b uint64
	for i := 0; i < 4; i++ {
		val := uint64(math.Float32bits(s[i]))
		if i < 2 {
			a |= val << (32 * i)
		} else {
			b |= val << (32 * (i - 2))
		}
	}
	return Float32s{a: a, b: b}
}

// LoadFloat32sPart loads a partial slice of float32 into an Float32s vector.
func LoadFloat32sPart(s []float32) (Float32s, int) {
	var a, b uint64
	n := len(s)
	if n > 4 {
		n = 4
	}
	for i := 0; i < n; i++ {
		val := uint64(math.Float32bits(s[i]))
		if i < 2 {
			a |= val << (32 * i)
		} else {
			b |= val << (32 * (i - 2))
		}
	}
	return Float32s{a: a, b: b}, n
}

func (x Float32s) get(i int) float32 {
	if i < 2 {
		return math.Float32frombits(uint32(x.a >> (32 * i)))
	}
	return math.Float32frombits(uint32(x.b >> (32 * (i - 2))))
}

func (x *Float32s) set(i int, v float32) {
	val := uint64(math.Float32bits(v))
	if i < 2 {
		mask := uint64(0xffffffff) << (32 * i)
		x.a = (x.a &^ mask) | (val << (32 * i))
	} else {
		mask := uint64(0xffffffff) << (32 * (i - 2))
		x.b = (x.b &^ mask) | (val << (32 * (i - 2)))
	}
}

// Abs returns the element-wise absolute value of x.
func (x Float32s) Abs() Float32s {
	var res Float32s
	for i := 0; i < 4; i++ {
		v := x.get(i)
		if v < 0 {
			res.set(i, -v)
		} else {
			res.set(i, v)
		}
	}
	return res
}

// Add returns the element-wise sum of x and y.
func (x Float32s) Add(y Float32s) Float32s {
	var res Float32s
	for i := 0; i < 4; i++ {
		res.set(i, x.get(i)+y.get(i))
	}
	return res
}

// ConvertToInt32 converts the vector elements to int32.
func (x Float32s) ConvertToInt32() Int32s {
	var res Int32s
	for i := 0; i < 4; i++ {
		res.set(i, int32(x.get(i)))
	}
	return res
}

// Div returns the element-wise quotient of x and y.
func (x Float32s) Div(y Float32s) Float32s {
	var res Float32s
	for i := 0; i < 4; i++ {
		res.set(i, x.get(i)/y.get(i))
	}
	return res
}

// Equal returns a mask indicating where x and y are equal.
func (x Float32s) Equal(y Float32s) Mask32s {
	var res Mask32s
	for i := 0; i < 4; i++ {
		if x.get(i) == y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// Greater returns a mask indicating where x is greater than y.
func (x Float32s) Greater(y Float32s) Mask32s {
	var res Mask32s
	for i := 0; i < 4; i++ {
		if x.get(i) > y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// GreaterEqual returns a mask indicating where x is greater than or equal to y.
func (x Float32s) GreaterEqual(y Float32s) Mask32s {
	var res Mask32s
	for i := 0; i < 4; i++ {
		if x.get(i) >= y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// Len returns the number of elements in the vector.
func (x Float32s) Len() int {
	return 4
}

// Less returns a mask indicating where x is less than y.
func (x Float32s) Less(y Float32s) Mask32s {
	var res Mask32s
	for i := 0; i < 4; i++ {
		if x.get(i) < y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// LessEqual returns a mask indicating where x is less than or equal to y.
func (x Float32s) LessEqual(y Float32s) Mask32s {
	var res Mask32s
	for i := 0; i < 4; i++ {
		if x.get(i) <= y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// Masked returns a new vector with elements from x where mask is true, and zero elsewhere.
func (x Float32s) Masked(mask Mask32s) Float32s {
	return Float32s{a: x.a & mask.a, b: x.b & mask.b}
}

// Max returns the element-wise maximum of x and y.
func (x Float32s) Max(y Float32s) Float32s {
	var res Float32s
	for i := 0; i < 4; i++ {
		vx := x.get(i)
		vy := y.get(i)
		if vx > vy {
			res.set(i, vx)
		} else {
			res.set(i, vy)
		}
	}
	return res
}

// IfElse returns a new vector with elements from x where mask is true, and y where mask is false.
func (x Float32s) IfElse(mask Mask32s, y Float32s) Float32s {
	return Float32s{
		a: (x.a & mask.a) | (y.a &^ mask.a),
		b: (x.b & mask.b) | (y.b &^ mask.b),
	}
}

// Min returns the element-wise minimum of x and y.
func (x Float32s) Min(y Float32s) Float32s {
	var res Float32s
	for i := 0; i < 4; i++ {
		vx := x.get(i)
		vy := y.get(i)
		if vx < vy {
			res.set(i, vx)
		} else {
			res.set(i, vy)
		}
	}
	return res
}

// Mul returns the element-wise product of x and y.
func (x Float32s) Mul(y Float32s) Float32s {
	var res Float32s
	for i := 0; i < 4; i++ {
		res.set(i, x.get(i)*y.get(i))
	}
	return res
}

// MulAdd returns x * y + z element-wise.
func (x Float32s) MulAdd(y, z Float32s) Float32s {
	var res Float32s
	for i := 0; i < 4; i++ {
		res.set(i, x.get(i)+y.get(i)*z.get(i))
	}
	return res
}

// Neg returns the element-wise negation of x.
func (x Float32s) Neg() Float32s {
	var res Float32s
	for i := 0; i < 4; i++ {
		res.set(i, -(x.get(i)))
	}
	return res
}

// NotEqual returns a mask indicating where x and y are not equal.
func (x Float32s) NotEqual(y Float32s) Mask32s {
	var res Mask32s
	for i := 0; i < 4; i++ {
		if x.get(i) != y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// Sqrt returns the element-wise square root of x.
func (x Float32s) Sqrt() Float32s {
	var res Float32s
	for i := 0; i < 4; i++ {
		res.set(i, float32(math.Sqrt(float64(x.get(i)))))
	}
	return res
}

// Store stores the vector elements into the slice s.
func (x Float32s) Store(s []float32) {
	for i := 0; i < 4 && i < len(s); i++ {
		s[i] = x.get(i)
	}
}

// StorePart stores a partial vector into the slice s.
func (x Float32s) StorePart(s []float32) {
	x.Store(s)
}

// String returns a string representation of the vector.
func (x Float32s) String() string {
	var parts [4]float32
	for i := 0; i < 4; i++ {
		parts[i] = x.get(i)
	}
	return fmt.Sprint(parts)
}

// Sub returns the element-wise difference of x and y.
func (x Float32s) Sub(y Float32s) Float32s {
	var res Float32s
	for i := 0; i < 4; i++ {
		res.set(i, x.get(i)-y.get(i))
	}
	return res
}

// ToBits reinterprets the vector bits as a Uint32s vector.
func (x Float32s) ToBits() Uint32s {
	return Uint32s{a: x.a, b: x.b}
}

// Float64s represents a 128-bit vector of 2 float64 elements.
type Float64s struct {
	_    _simd
	a, b uint64
}

// LoadFloat64s loads a slice of float64 into an Float64s vector.
func LoadFloat64s(s []float64) Float64s {
	var a, b uint64
	a = math.Float64bits(s[0])
	b = math.Float64bits(s[1])
	return Float64s{a: a, b: b}
}

// LoadFloat64sPart loads a partial slice of float64 into an Float64s vector.
func LoadFloat64sPart(s []float64) (Float64s, int) {
	n := len(s)
	var a, b uint64
	if n > 0 {
		a = math.Float64bits(s[0])
	}
	if n > 1 {
		b = math.Float64bits(s[1])
	}
	return Float64s{a: a, b: b}, n
}

func (x Float64s) get(i int) float64 {
	if i == 0 {
		return math.Float64frombits(x.a)
	}
	return math.Float64frombits(x.b)
}

func (x *Float64s) set(i int, v float64) {
	if i == 0 {
		x.a = math.Float64bits(v)
	} else {
		x.b = math.Float64bits(v)
	}
}

// Abs returns the element-wise absolute value of x.
func (x Float64s) Abs() Float64s {
	var res Float64s
	for i := 0; i < 4; i++ {
		v := x.get(i)
		if v < 0 {
			res.set(i, -v)
		} else {
			res.set(i, v)
		}
	}
	return res
}

// Add returns the element-wise sum of x and y.
func (x Float64s) Add(y Float64s) Float64s {
	var res Float64s
	res.set(0, x.get(0)+y.get(0))
	res.set(1, x.get(1)+y.get(1))
	return res
}

// Div returns the element-wise quotient of x and y.
func (x Float64s) Div(y Float64s) Float64s {
	var res Float64s
	res.set(0, x.get(0)/y.get(0))
	res.set(1, x.get(1)/y.get(1))
	return res
}

// Equal returns a mask indicating where x and y are equal.
func (x Float64s) Equal(y Float64s) Mask64s {
	var res Mask64s
	if x.get(0) == y.get(0) {
		res.a = ^uint64(0)
	}
	if x.get(1) == y.get(1) {
		res.b = ^uint64(0)
	}
	return res
}

// Greater returns a mask indicating where x is greater than y.
func (x Float64s) Greater(y Float64s) Mask64s {
	var res Mask64s
	if x.get(0) > y.get(0) {
		res.a = ^uint64(0)
	}
	if x.get(1) > y.get(1) {
		res.b = ^uint64(0)
	}
	return res
}

// GreaterEqual returns a mask indicating where x is greater than or equal to y.
func (x Float64s) GreaterEqual(y Float64s) Mask64s {
	var res Mask64s
	if x.get(0) >= y.get(0) {
		res.a = ^uint64(0)
	}
	if x.get(1) >= y.get(1) {
		res.b = ^uint64(0)
	}
	return res
}

// Len returns the number of elements in the vector.
func (x Float64s) Len() int {
	return 2
}

// Less returns a mask indicating where x is less than y.
func (x Float64s) Less(y Float64s) Mask64s {
	var res Mask64s
	if x.get(0) < y.get(0) {
		res.a = ^uint64(0)
	}
	if x.get(1) < y.get(1) {
		res.b = ^uint64(0)
	}
	return res
}

// LessEqual returns a mask indicating where x is less than or equal to y.
func (x Float64s) LessEqual(y Float64s) Mask64s {
	var res Mask64s
	if x.get(0) <= y.get(0) {
		res.a = ^uint64(0)
	}
	if x.get(1) <= y.get(1) {
		res.b = ^uint64(0)
	}
	return res
}

// Masked returns a new vector with elements from x where mask is true, and zero elsewhere.
func (x Float64s) Masked(mask Mask64s) Float64s {
	return Float64s{a: x.a & mask.a, b: x.b & mask.b}
}

// Max returns the element-wise maximum of x and y.
func (x Float64s) Max(y Float64s) Float64s {
	var res Float64s
	vx := x.get(0)
	vy := y.get(0)
	if vx > vy {
		res.set(0, vx)
	} else {
		res.set(0, vy)
	}
	vx = x.get(1)
	vy = y.get(1)
	if vx > vy {
		res.set(1, vx)
	} else {
		res.set(1, vy)
	}
	return res
}

// IfElse returns a new vector with elements from x where mask is true, and y where mask is false.
func (x Float64s) IfElse(mask Mask64s, y Float64s) Float64s {
	return Float64s{
		a: (x.a & mask.a) | (y.a &^ mask.a),
		b: (x.b & mask.b) | (y.b &^ mask.b),
	}
}

// Min returns the element-wise minimum of x and y.
func (x Float64s) Min(y Float64s) Float64s {
	var res Float64s
	vx := x.get(0)
	vy := y.get(0)
	if vx < vy {
		res.set(0, vx)
	} else {
		res.set(0, vy)
	}
	vx = x.get(1)
	vy = y.get(1)
	if vx < vy {
		res.set(1, vx)
	} else {
		res.set(1, vy)
	}
	return res
}

// Mul returns the element-wise product of x and y.
func (x Float64s) Mul(y Float64s) Float64s {
	var res Float64s
	res.set(0, x.get(0)*y.get(0))
	res.set(1, x.get(1)*y.get(1))
	return res
}

// MulAdd returns x * y + z element-wise.
func (x Float64s) MulAdd(y, z Float64s) Float64s {
	var res Float64s
	res.set(0, x.get(0)+y.get(0)*z.get(0))
	res.set(1, x.get(1)+y.get(1)*z.get(1))
	return res
}

// Neg returns the element-wise negation of x.
func (x Float64s) Neg() Float64s {
	var res Float64s
	for i := 0; i < 4; i++ {
		res.set(i, -(x.get(i)))
	}
	return res
}

// NotEqual returns a mask indicating where x and y are not equal.
func (x Float64s) NotEqual(y Float64s) Mask64s {
	var res Mask64s
	if x.get(0) != y.get(0) {
		res.a = ^uint64(0)
	}
	if x.get(1) != y.get(1) {
		res.b = ^uint64(0)
	}
	return res
}

// Sqrt returns the element-wise square root of x.
func (x Float64s) Sqrt() Float64s {
	var res Float64s
	res.set(0, math.Sqrt(x.get(0)))
	res.set(1, math.Sqrt(x.get(1)))
	return res
}

// Store stores the vector elements into the slice s.
func (x Float64s) Store(s []float64) {
	if len(s) > 0 {
		s[0] = x.get(0)
	}
	if len(s) > 1 {
		s[1] = x.get(1)
	}
}

// StorePart stores a partial vector into the slice s.
func (x Float64s) StorePart(s []float64) {
	x.Store(s)
}

// String returns a string representation of the vector.
func (x Float64s) String() string {
	return fmt.Sprint([2]float64{x.get(0), x.get(1)})
}

// Sub returns the element-wise difference of x and y.
func (x Float64s) Sub(y Float64s) Float64s {
	var res Float64s
	res.set(0, x.get(0)-y.get(0))
	res.set(1, x.get(1)-y.get(1))
	return res
}

// ToBits reinterprets the vector bits as a Uint64s vector.
func (x Float64s) ToBits() Uint64s {
	return Uint64s{a: x.a, b: x.b}
}

// Mask8s represents a 128-bit mask vector for 16 int8/uint8 elements.
type Mask8s struct {
	_    _simd
	a, b uint64
}

func (x *Mask8s) set(i int, v bool) {
	if v {
		if i < 8 {
			mask := uint64(0xff) << (8 * i)
			x.a |= mask
		} else {
			mask := uint64(0xff) << (8 * (i - 8))
			x.b |= mask
		}
	}
}

// And returns the bitwise AND of x and y.
func (x Mask8s) And(y Mask8s) Mask8s {
	return Mask8s{a: x.a & y.a, b: x.b & y.b}
}

// Or returns the bitwise OR of x and y.
func (x Mask8s) Or(y Mask8s) Mask8s {
	return Mask8s{a: x.a | y.a, b: x.b | y.b}
}

// String returns a string representation of the vector.
func (x Mask8s) String() string {
	return fmt.Sprintf("{a:%#x, b:%#x}", x.a, x.b)
}

// ToInt8s converts the mask to an Int8s vector.
func (x Mask8s) ToInt8s() Int8s {
	return Int8s{a: x.a, b: x.b}
}

// Mask16s represents a 128-bit mask vector for 8 int16/uint16 elements.
type Mask16s struct {
	_    _simd
	a, b uint64
}

func (x *Mask16s) set(i int, v bool) {
	if v {
		if i < 4 {
			mask := uint64(0xffff) << (16 * i)
			x.a |= mask
		} else {
			mask := uint64(0xffff) << (16 * (i - 4))
			x.b |= mask
		}
	}
}

// And returns the bitwise AND of x and y.
func (x Mask16s) And(y Mask16s) Mask16s {
	return Mask16s{a: x.a & y.a, b: x.b & y.b}
}

// Or returns the bitwise OR of x and y.
func (x Mask16s) Or(y Mask16s) Mask16s {
	return Mask16s{a: x.a | y.a, b: x.b | y.b}
}

// String returns a string representation of the vector.
func (x Mask16s) String() string {
	return fmt.Sprintf("{a:%#x, b:%#x}", x.a, x.b)
}

// ToInt16s converts the mask to an Int16s vector.
func (x Mask16s) ToInt16s() Int16s {
	return Int16s{a: x.a, b: x.b}
}

// Mask32s represents a 128-bit mask vector for 4 int32/uint32/float32 elements.
type Mask32s struct {
	_    _simd
	a, b uint64
}

func (x *Mask32s) set(i int, v bool) {
	if v {
		if i < 2 {
			mask := uint64(0xffffffff) << (32 * i)
			x.a |= mask
		} else {
			mask := uint64(0xffffffff) << (32 * (i - 2))
			x.b |= mask
		}
	}
}

// And returns the bitwise AND of x and y.
func (x Mask32s) And(y Mask32s) Mask32s {
	return Mask32s{a: x.a & y.a, b: x.b & y.b}
}

// Or returns the bitwise OR of x and y.
func (x Mask32s) Or(y Mask32s) Mask32s {
	return Mask32s{a: x.a | y.a, b: x.b | y.b}
}

// String returns a string representation of the vector.
func (x Mask32s) String() string {
	return fmt.Sprintf("{a:%#x, b:%#x}", x.a, x.b)
}

// ToInt32s converts the mask to an Int32s vector.
func (x Mask32s) ToInt32s() Int32s {
	return Int32s{a: x.a, b: x.b}
}

// Mask64s represents a 128-bit mask vector for 2 int64/uint64/float64 elements.
type Mask64s struct {
	_    _simd
	a, b uint64
}

func (x *Mask64s) set(i int, v bool) {
	if v {
		if i == 0 {
			x.a = ^uint64(0)
		} else {
			x.b = ^uint64(0)
		}
	}
}

// And returns the bitwise AND of x and y.
func (x Mask64s) And(y Mask64s) Mask64s {
	return Mask64s{a: x.a & y.a, b: x.b & y.b}
}

// Or returns the bitwise OR of x and y.
func (x Mask64s) Or(y Mask64s) Mask64s {
	return Mask64s{a: x.a | y.a, b: x.b | y.b}
}

// String returns a string representation of the vector.
func (x Mask64s) String() string {
	return fmt.Sprintf("{a:%#x, b:%#x}", x.a, x.b)
}

// ToInt64s converts the mask to an Int64s vector.
func (x Mask64s) ToInt64s() Int64s {
	return Int64s{a: x.a, b: x.b}
}

func newT(lo, hi uint64) Uint64s {
	return Uint64s{a: lo, b: hi}
}

// mwl returns the 128-bit product of the lower halves of x and y
func (x Uint64s) mwl(y Uint64s) Uint64s {
	hi, lo := bits.Mul64(x.a, y.a)
	return Uint64s{a: lo, b: hi}
}

var (
	// For mK, bits J such that J mod 5 == K are set
	m0 = newT(0x0084210842108421, 0x1108421084210842)
	m1 = newT(0x1108421084210842, 0x3210842108421084)
	m2 = newT(0x3210842108421084, 0x8421084210842108)
	m3 = newT(0x8421084210842108, 0x0842108421084210)
	m4 = newT(0x0842108421084210, 0x0084210842108421)
)

func (x Uint64s) clmul(y Uint64s) Uint64s {
	x0 := x.And(m0)
	x1 := x.And(m1)
	x2 := x.And(m2)
	x3 := x.And(m3)
	x4 := x.And(m4)

	y0 := y.And(m0)
	y1 := y.And(m1)
	y2 := y.And(m2)
	y3 := y.And(m3)
	y4 := y.And(m4)

	// sum of x, y indices == K mod 5; mask index = K
	z := (x0.mwl(y0)).Xor(x1.mwl(y4)).Xor(x4.mwl(y1)).Xor(x2.mwl(y3)).Xor(x3.mwl(y2)).And(m0)
	z = (x3.mwl(y3)).Xor(x2.mwl(y4)).Xor(x4.mwl(y2)).Xor(x0.mwl(y1)).Xor(x1.mwl(y0)).And(m1).Or(z)
	z = (x1.mwl(y1)).Xor(x3.mwl(y4)).Xor(x4.mwl(y3)).Xor(x0.mwl(y2)).Xor(x2.mwl(y0)).And(m2).Or(z)
	z = (x4.mwl(y4)).Xor(x0.mwl(y3)).Xor(x3.mwl(y0)).Xor(x1.mwl(y2)).Xor(x2.mwl(y1)).And(m3).Or(z)
	z = (x2.mwl(y2)).Xor(x0.mwl(y4)).Xor(x4.mwl(y0)).Xor(x1.mwl(y3)).Xor(x3.mwl(y1)).And(m4).Or(z)

	return z
}

// CarrylessMultiplyEven computes the carryless
// multiplications of selected even halves of the elements of x and y.
// The result fills the 128 bits of each even-odd pair.
//
// A carryless multiplication uses bitwise XOR instead of
// add-with-carry, for example (in base two):
//
//	11 * 11 = 11 * (10 ^ 1) = (11 * 10) ^ (11 * 1) = 110 ^ 11 = 101
//
// This also models multiplication of polynomials with coefficients
// from GF(2) -- 11 * 11 models (x+1)*(x+1) = x**2 + (1^1)x + 1 =
// x**2 + 0x + 1 = x**2 + 1 modeled by 101.  (Note that "+" adds
// polynomial terms, but coefficients "add" with XOR.)
func (x Uint64s) CarrylessMultiplyEven(y Uint64s) Uint64s {
	return x.clmul(y)
}

// CarrylessMultiplyOdd computes the carryless
// multiplications of selected odd halves of the elements of x and y.
// The result fills the 128 bits of each even-odd pair.
//
// A carryless multiplication uses bitwise XOR instead of
// add-with-carry, for example (in base two):
//
//	11 * 11 = 11 * (10 ^ 1) = (11 * 10) ^ (11 * 1) = 110 ^ 11 = 101
//
// This also models multiplication of polynomials with coefficients
// from GF(2) -- 11 * 11 models (x+1)*(x+1) = x**2 + (1^1)x + 1 =
// x**2 + 0x + 1 = x**2 + 1 modeled by 101.  (Note that "+" adds
// polynomial terms, but coefficients "add" with XOR.)
func (x Uint64s) CarrylessMultiplyOdd(y Uint64s) Uint64s {
	x.a = x.b
	y.a = y.b
	return x.clmul(y)
}
