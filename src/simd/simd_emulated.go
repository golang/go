// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && !(amd64 || wasm || arm64)

package simd

import (
	"internal/strconv"
	"math"
	"math/bits"
)

// VectorBitSize returns the bit length of the emulated vector (fixed to 128).
func VectorBitSize() int {
	return 128
}

// Emulated returns whether simd is emulated.
func Emulated() bool {
	return true
}

// HasHardwareCarrylessMultiply returns whether this platform
// has a hardware-implemented version of carryless multiply.
// With default GODEBUG=simd settings, if this is false,
// it is emulated and merely slow, but with non-default settings
// this can indicate the possibility of a missing instruction
// that will fail ("SIGILL") if it is executed.
func HasHardwareCarrylessMultiply() bool {
	return false
}

type number interface {
	~int8 | ~int16 | ~int32 | ~int64 | ~uint8 | ~uint16 | ~uint32 | ~uint64 | ~float32 | ~float64
}

func sliceToString[T number](x []T) string {
	s := ""
	pfx := "{"
	for _, y := range x {
		s += pfx
		pfx = ","
		switch e := any(y).(type) {
		case int8:
			s += strconv.Itoa(int(e))
		case int16:
			s += strconv.Itoa(int(e))
		case int32:
			s += strconv.Itoa(int(e))
		case int64:
			s += strconv.FormatInt(int64(e), 10)
		case uint8:
			s += strconv.FormatUint(uint64(e), 10)
		case uint16:
			s += strconv.FormatUint(uint64(e), 10)
		case uint32:
			s += strconv.FormatUint(uint64(e), 10)
		case uint64:
			s += strconv.FormatUint(uint64(e), 10)
		case float32:
			s += strconv.FormatFloat(float64(e), 'g', -1, 32)
		case float64:
			s += strconv.FormatFloat(e, 'g', -1, 64)
		}
	}
	s += "}"
	return s
}

// LoadInt8s loads a slice into a vector. If len(s) < z.Len(), it panics.
func LoadInt8s(s []int8) (z Int8s) {
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

// LoadInt8sPart loads n=min(len(s), z.Len()) elements from slice s as a
// vector and returns the vector and n. If len(s) < z.Len(), the
// remaining vector elements will be zero.
func LoadInt8sPart(s []int8) (z Int8s, n int) {
	var a, b uint64
	n = len(s)
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

// Abs returns the elementwise absolute value of x.
func (x Int8s) Abs() (z Uint8s) {
	var res Uint8s
	for i := 0; i < 16; i++ {
		v := x.get(i)
		if v < 0 {
			res.set(i, uint8(-v))
		} else {
			res.set(i, uint8(v))
		}
	}
	return res
}

// Add adds x and y elementwise.
//
//	z[i] = x[i] + y[i]
func (x Int8s) Add(y Int8s) (z Int8s) {
	var res Int8s
	for i := 0; i < 16; i++ {
		res.set(i, x.get(i)+y.get(i))
	}
	return res
}

// AddSaturated adds x and y elementwise with saturation.
//
//	z[i] = sat(x[i] + y[i])
func (x Int8s) AddSaturated(y Int8s) (z Int8s) {
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
//
//	z[i] = x[i] & y[i]
func (x Int8s) And(y Int8s) (z Int8s) {
	return Int8s{a: x.a & y.a, b: x.b & y.b}
}

// AndNot returns the bitwise AND NOT of x and y.
//
//	z[i] = x[i] &^ y[i]
func (x Int8s) AndNot(y Int8s) (z Int8s) {
	return Int8s{a: x.a &^ y.a, b: x.b &^ y.b}
}

// Equal returns a mask indicating which elements of x and y are equal.
func (x Int8s) Equal(y Int8s) (z Mask8s) {
	var res Mask8s
	for i := 0; i < 16; i++ {
		if x.get(i) == y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// Greater returns a mask indicating which elements of x are greater than y.
//
//	z[i] = x[i] > y[i]
func (x Int8s) Greater(y Int8s) (z Mask8s) {
	var res Mask8s
	for i := 0; i < 16; i++ {
		if x.get(i) > y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// GreaterEqual returns a mask indicating which elements of x are greater than
// or equal to y.
//
//	z[i] = x[i] >= y[i]
func (x Int8s) GreaterEqual(y Int8s) (z Mask8s) {
	var res Mask8s
	for i := 0; i < 16; i++ {
		if x.get(i) >= y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// Less returns a mask indicating which elements of x are less than y.
//
//	z[i] = x[i] < y[i]
func (x Int8s) Less(y Int8s) (z Mask8s) {
	var res Mask8s
	for i := 0; i < 16; i++ {
		if x.get(i) < y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// LessEqual returns a mask indicating which elements of x are less than or
// equal to y.
//
//	z[i] = x[i] <= y[i]
func (x Int8s) LessEqual(y Int8s) (z Mask8s) {
	var res Mask8s
	for i := 0; i < 16; i++ {
		if x.get(i) <= y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// NotEqual returns a mask indicating which elements of x and y are not equal.
func (x Int8s) NotEqual(y Int8s) (z Mask8s) {
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

// Masked returns a vector with elements from x where mask is true, and zero
// elsewhere.
//
//	z[i] = if mask[i] { x[i] } else { 0 }
func (x Int8s) Masked(mask Mask8s) (z Int8s) {
	return Int8s{a: x.a & mask.a, b: x.b & mask.b}
}

// Max returns the elementwise maximum of x and y.
//
//	z[i] = max(x[i], y[i])
func (x Int8s) Max(y Int8s) (z Int8s) {
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

// Mul returns the elementwise product of x and y.
//
//	z[i] = x[i] * y[i]
func (x Int8s) Mul(y Int8s) (z Int8s) {
	var res Int8s
	for i := 0; i < 16; i++ {
		res.set(i, x.get(i)*y.get(i))
	}
	return res
}

// IfElse returns a vector with elements from x where mask is true, and y where
// mask is false.
//
//	z[i] = if mask[i] { x[i] } else { y[i] }
func (x Int8s) IfElse(mask Mask8s, y Int8s) (z Int8s) {
	return Int8s{
		a: (x.a & mask.a) | (y.a &^ mask.a),
		b: (x.b & mask.b) | (y.b &^ mask.b),
	}
}

// Min returns the elementwise minimum of x and y.
//
//	z[i] = min(x[i], y[i])
func (x Int8s) Min(y Int8s) (z Int8s) {
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

// Neg returns the elementwise negation of x.
//
//	z[i] = -x[i]
func (x Int8s) Neg() (z Int8s) {
	var res Int8s
	for i := 0; i < 16; i++ {
		res.set(i, -x.get(i))
	}
	return res
}

// Not returns the bitwise negation of x.
//
//	z[i] = ^x[i]
func (x Int8s) Not() (z Int8s) {
	return Int8s{a: ^x.a, b: ^x.b}
}

// Or returns the bitwise OR of x and y.
//
//	z[i] = x[i] | y[i]
func (x Int8s) Or(y Int8s) (z Int8s) {
	return Int8s{a: x.a | y.a, b: x.b | y.b}
}

// ReduceSum returns the scalar sum of the elements of x.
//
//	z = x[0] + x[1] + ...
func (x Int8s) ReduceSum() (z int8) {
	var res int8
	for i := 0; i < 16; i++ {
		res += x.get(i)
	}
	return res
}

// Store stores the elements of x into a slice. If len(s) < x.Len(), it panics.
func (x Int8s) Store(s []int8) {
	for i := 0; i < 16 && i < len(s); i++ {
		s[i] = x.get(i)
	}
}

// StorePart stores n=min(len(s), x.Len()) elements of x into s and returns n.
func (x Int8s) StorePart(s []int8) (n int) {
	x.Store(s)
	return min(len(s), x.Len())
}

// String returns a string representation of the vector.
func (x Int8s) String() string {
	var parts [16]int8
	for i := 0; i < 16; i++ {
		parts[i] = x.get(i)
	}
	return sliceToString(parts[:])
}

// Sub subtracts y from x elementwise.
//
//	z[i] = x[i] - y[i]
func (x Int8s) Sub(y Int8s) (z Int8s) {
	var res Int8s
	for i := 0; i < 16; i++ {
		res.set(i, x.get(i)-y.get(i))
	}
	return res
}

// SubSaturated subtracts y from x elementwise with saturation.
//
//	z[i] = sat(x[i] - y[i])
func (x Int8s) SubSaturated(y Int8s) (z Int8s) {
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

// ToMask returns a mask indicating which elements of x are non-zero.
//
//	z[i] = x[i] != 0
func (x Int8s) ToMask() (z Mask8s) {
	var res Mask8s
	for i := 0; i < 16; i++ {
		if x.get(i) != 0 {
			res.set(i, true)
		}
	}
	return res
}

// Xor returns the bitwise XOR of x and y.
//
//	z[i] = x[i] ^ y[i]
func (x Int8s) Xor(y Int8s) (z Int8s) {
	return Int8s{a: x.a ^ y.a, b: x.b ^ y.b}
}

// ConvertToUint8 converts each element of x to uint8.
func (x Int8s) ConvertToUint8() (z Uint8s) {
	return Uint8s{a: x.a, b: x.b}
}

// ToBits reinterprets the bits of each element of x as type uint8.
func (x Int8s) ToBits() (z Uint8s) {
	return Uint8s{a: x.a, b: x.b}
}

// LoadInt16s loads a slice into a vector. If len(s) < z.Len(), it panics.
func LoadInt16s(s []int16) (z Int16s) {
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

// LoadInt16sPart loads n=min(len(s), z.Len()) elements from slice s as a
// vector and returns the vector and n. If len(s) < z.Len(), the
// remaining vector elements will be zero.
func LoadInt16sPart(s []int16) (z Int16s, n int) {
	var a, b uint64
	n = len(s)
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

// Abs returns the elementwise absolute value of x.
func (x Int16s) Abs() (z Uint16s) {
	var res Uint16s
	for i := 0; i < 8; i++ {
		v := x.get(i)
		if v < 0 {
			res.set(i, uint16(-v))
		} else {
			res.set(i, uint16(v))
		}
	}
	return res
}

// Add adds x and y elementwise.
//
//	z[i] = x[i] + y[i]
func (x Int16s) Add(y Int16s) (z Int16s) {
	var res Int16s
	for i := 0; i < 8; i++ {
		res.set(i, x.get(i)+y.get(i))
	}
	return res
}

// AddSaturated adds x and y elementwise with saturation.
//
//	z[i] = sat(x[i] + y[i])
func (x Int16s) AddSaturated(y Int16s) (z Int16s) {
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
//
//	z[i] = x[i] & y[i]
func (x Int16s) And(y Int16s) (z Int16s) {
	return Int16s{a: x.a & y.a, b: x.b & y.b}
}

// AndNot returns the bitwise AND NOT of x and y.
//
//	z[i] = x[i] &^ y[i]
func (x Int16s) AndNot(y Int16s) (z Int16s) {
	return Int16s{a: x.a &^ y.a, b: x.b &^ y.b}
}

// Equal returns a mask indicating which elements of x and y are equal.
func (x Int16s) Equal(y Int16s) (z Mask16s) {
	var res Mask16s
	for i := 0; i < 8; i++ {
		if x.get(i) == y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// Greater returns a mask indicating which elements of x are greater than y.
//
//	z[i] = x[i] > y[i]
func (x Int16s) Greater(y Int16s) (z Mask16s) {
	var res Mask16s
	for i := 0; i < 8; i++ {
		if x.get(i) > y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// GreaterEqual returns a mask indicating which elements of x are greater than
// or equal to y.
//
//	z[i] = x[i] >= y[i]
func (x Int16s) GreaterEqual(y Int16s) (z Mask16s) {
	var res Mask16s
	for i := 0; i < 8; i++ {
		if x.get(i) >= y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// Less returns a mask indicating which elements of x are less than y.
//
//	z[i] = x[i] < y[i]
func (x Int16s) Less(y Int16s) (z Mask16s) {
	var res Mask16s
	for i := 0; i < 8; i++ {
		if x.get(i) < y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// LessEqual returns a mask indicating which elements of x are less than or
// equal to y.
//
//	z[i] = x[i] <= y[i]
func (x Int16s) LessEqual(y Int16s) (z Mask16s) {
	var res Mask16s
	for i := 0; i < 8; i++ {
		if x.get(i) <= y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// NotEqual returns a mask indicating which elements of x and y are not equal.
func (x Int16s) NotEqual(y Int16s) (z Mask16s) {
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

// Masked returns a vector with elements from x where mask is true, and zero
// elsewhere.
//
//	z[i] = if mask[i] { x[i] } else { 0 }
func (x Int16s) Masked(mask Mask16s) (z Int16s) {
	return Int16s{a: x.a & mask.a, b: x.b & mask.b}
}

// Max returns the elementwise maximum of x and y.
//
//	z[i] = max(x[i], y[i])
func (x Int16s) Max(y Int16s) (z Int16s) {
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

// IfElse returns a vector with elements from x where mask is true, and y where
// mask is false.
//
//	z[i] = if mask[i] { x[i] } else { y[i] }
func (x Int16s) IfElse(mask Mask16s, y Int16s) (z Int16s) {
	return Int16s{
		a: (x.a & mask.a) | (y.a &^ mask.a),
		b: (x.b & mask.b) | (y.b &^ mask.b),
	}
}

// Min returns the elementwise minimum of x and y.
//
//	z[i] = min(x[i], y[i])
func (x Int16s) Min(y Int16s) (z Int16s) {
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

// Mul returns the elementwise product of x and y.
//
//	z[i] = x[i] * y[i]
func (x Int16s) Mul(y Int16s) (z Int16s) {
	var res Int16s
	for i := 0; i < 8; i++ {
		res.set(i, x.get(i)*y.get(i))
	}
	return res
}

// Neg returns the elementwise negation of x.
//
//	z[i] = -x[i]
func (x Int16s) Neg() (z Int16s) {
	var res Int16s
	for i := 0; i < 8; i++ {
		res.set(i, -x.get(i))
	}
	return res
}

// Not returns the bitwise negation of x.
//
//	z[i] = ^x[i]
func (x Int16s) Not() (z Int16s) {
	return Int16s{a: ^x.a, b: ^x.b}
}

// Or returns the bitwise OR of x and y.
//
//	z[i] = x[i] | y[i]
func (x Int16s) Or(y Int16s) (z Int16s) {
	return Int16s{a: x.a | y.a, b: x.b | y.b}
}

// ShiftAllLeft shifts each element of x left by shift bits.
// If shift is greater than the element width, the result is 0.
//
//	z[i] = x[i] << shift
func (x Int16s) ShiftAllLeft(shift uint64) (z Int16s) {
	var res Int16s
	for i := 0; i < 8; i++ {
		res.set(i, x.get(i)<<shift)
	}
	return res
}

// ShiftAllRight arithmetically shifts each element of x right by shift bits.
// If shift is greater than the element width, the result is 0 or -1.
//
//	z[i] = x[i] >> shift
func (x Int16s) ShiftAllRight(shift uint64) (z Int16s) {
	var res Int16s
	for i := 0; i < 8; i++ {
		res.set(i, x.get(i)>>shift)
	}
	return res
}

// ReduceSum returns the scalar sum of the elements of x.
//
//	z = x[0] + x[1] + ...
func (x Int16s) ReduceSum() (z int16) {
	var res int16
	for i := 0; i < 8; i++ {
		res += x.get(i)
	}
	return res
}

// Store stores the elements of x into a slice. If len(s) < x.Len(), it panics.
func (x Int16s) Store(s []int16) {
	for i := 0; i < 8 && i < len(s); i++ {
		s[i] = x.get(i)
	}
}

// StorePart stores n=min(len(s), x.Len()) elements of x into s and returns n.
func (x Int16s) StorePart(s []int16) (n int) {
	x.Store(s)
	return min(len(s), x.Len())
}

// String returns a string representation of the vector.
func (x Int16s) String() string {
	var parts [8]int16
	for i := 0; i < 8; i++ {
		parts[i] = x.get(i)
	}
	return sliceToString(parts[:])
}

// Sub subtracts y from x elementwise.
//
//	z[i] = x[i] - y[i]
func (x Int16s) Sub(y Int16s) (z Int16s) {
	var res Int16s
	for i := 0; i < 8; i++ {
		res.set(i, x.get(i)-y.get(i))
	}
	return res
}

// SubSaturated subtracts y from x elementwise with saturation.
//
//	z[i] = sat(x[i] - y[i])
func (x Int16s) SubSaturated(y Int16s) (z Int16s) {
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

// ToMask returns a mask indicating which elements of x are non-zero.
//
//	z[i] = x[i] != 0
func (x Int16s) ToMask() (z Mask16s) {
	var res Mask16s
	for i := 0; i < 8; i++ {
		if x.get(i) != 0 {
			res.set(i, true)
		}
	}
	return res
}

// Xor returns the bitwise XOR of x and y.
//
//	z[i] = x[i] ^ y[i]
func (x Int16s) Xor(y Int16s) (z Int16s) {
	return Int16s{a: x.a ^ y.a, b: x.b ^ y.b}
}

// ConvertToUint16 converts each element of x to uint16.
func (x Int16s) ConvertToUint16() (z Uint16s) {
	return Uint16s{a: x.a, b: x.b}
}

// ToBits reinterprets the bits of each element of x as type uint16.
func (x Int16s) ToBits() (z Uint16s) {
	return Uint16s{a: x.a, b: x.b}
}

// LoadInt32s loads a slice into a vector. If len(s) < z.Len(), it panics.
func LoadInt32s(s []int32) (z Int32s) {
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

// LoadInt32sPart loads n=min(len(s), z.Len()) elements from slice s as a
// vector and returns the vector and n. If len(s) < z.Len(), the
// remaining vector elements will be zero.
func LoadInt32sPart(s []int32) (z Int32s, n int) {
	var a, b uint64
	n = len(s)
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
		mask := uint64(0xffff_ffff) << (32 * i)
		x.a = (x.a &^ mask) | (val << (32 * i))
	} else {
		mask := uint64(0xffff_ffff) << (32 * (i - 2))
		x.b = (x.b &^ mask) | (val << (32 * (i - 2)))
	}
}

// Abs returns the elementwise absolute value of x.
func (x Int32s) Abs() (z Uint32s) {
	var res Uint32s
	for i := 0; i < 4; i++ {
		v := x.get(i)
		if v < 0 {
			res.set(i, uint32(-v))
		} else {
			res.set(i, uint32(v))
		}
	}
	return res
}

// Add adds x and y elementwise.
//
//	z[i] = x[i] + y[i]
func (x Int32s) Add(y Int32s) (z Int32s) {
	var res Int32s
	for i := 0; i < 4; i++ {
		res.set(i, x.get(i)+y.get(i))
	}
	return res
}

// And returns the bitwise AND of x and y.
//
//	z[i] = x[i] & y[i]
func (x Int32s) And(y Int32s) (z Int32s) {
	return Int32s{a: x.a & y.a, b: x.b & y.b}
}

// AndNot returns the bitwise AND NOT of x and y.
//
//	z[i] = x[i] &^ y[i]
func (x Int32s) AndNot(y Int32s) (z Int32s) {
	return Int32s{a: x.a &^ y.a, b: x.b &^ y.b}
}

// ConvertToFloat32 converts each element of x to float32.
func (x Int32s) ConvertToFloat32() (z Float32s) {
	var res Float32s
	for i := 0; i < 4; i++ {
		res.set(i, float32(x.get(i)))
	}
	return res
}

// Equal returns a mask indicating which elements of x and y are equal.
func (x Int32s) Equal(y Int32s) (z Mask32s) {
	var res Mask32s
	for i := 0; i < 4; i++ {
		if x.get(i) == y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// Greater returns a mask indicating which elements of x are greater than y.
//
//	z[i] = x[i] > y[i]
func (x Int32s) Greater(y Int32s) (z Mask32s) {
	var res Mask32s
	for i := 0; i < 4; i++ {
		if x.get(i) > y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// GreaterEqual returns a mask indicating which elements of x are greater than
// or equal to y.
//
//	z[i] = x[i] >= y[i]
func (x Int32s) GreaterEqual(y Int32s) (z Mask32s) {
	var res Mask32s
	for i := 0; i < 4; i++ {
		if x.get(i) >= y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// Less returns a mask indicating which elements of x are less than y.
//
//	z[i] = x[i] < y[i]
func (x Int32s) Less(y Int32s) (z Mask32s) {
	var res Mask32s
	for i := 0; i < 4; i++ {
		if x.get(i) < y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// LessEqual returns a mask indicating which elements of x are less than or
// equal to y.
//
//	z[i] = x[i] <= y[i]
func (x Int32s) LessEqual(y Int32s) (z Mask32s) {
	var res Mask32s
	for i := 0; i < 4; i++ {
		if x.get(i) <= y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// NotEqual returns a mask indicating which elements of x and y are not equal.
func (x Int32s) NotEqual(y Int32s) (z Mask32s) {
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

// Masked returns a vector with elements from x where mask is true, and zero
// elsewhere.
//
//	z[i] = if mask[i] { x[i] } else { 0 }
func (x Int32s) Masked(mask Mask32s) (z Int32s) {
	return Int32s{a: x.a & mask.a, b: x.b & mask.b}
}

// Max returns the elementwise maximum of x and y.
//
//	z[i] = max(x[i], y[i])
func (x Int32s) Max(y Int32s) (z Int32s) {
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

// IfElse returns a vector with elements from x where mask is true, and y where
// mask is false.
//
//	z[i] = if mask[i] { x[i] } else { y[i] }
func (x Int32s) IfElse(mask Mask32s, y Int32s) (z Int32s) {
	return Int32s{
		a: (x.a & mask.a) | (y.a &^ mask.a),
		b: (x.b & mask.b) | (y.b &^ mask.b),
	}
}

// Min returns the elementwise minimum of x and y.
//
//	z[i] = min(x[i], y[i])
func (x Int32s) Min(y Int32s) (z Int32s) {
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

// Mul returns the elementwise product of x and y.
//
//	z[i] = x[i] * y[i]
func (x Int32s) Mul(y Int32s) (z Int32s) {
	var res Int32s
	for i := 0; i < 4; i++ {
		res.set(i, x.get(i)*y.get(i))
	}
	return res
}

// Neg returns the elementwise negation of x.
//
//	z[i] = -x[i]
func (x Int32s) Neg() (z Int32s) {
	var res Int32s
	for i := 0; i < 4; i++ {
		res.set(i, -x.get(i))
	}
	return res
}

// Not returns the bitwise negation of x.
//
//	z[i] = ^x[i]
func (x Int32s) Not() (z Int32s) {
	return Int32s{a: ^x.a, b: ^x.b}
}

// Or returns the bitwise OR of x and y.
//
//	z[i] = x[i] | y[i]
func (x Int32s) Or(y Int32s) (z Int32s) {
	return Int32s{a: x.a | y.a, b: x.b | y.b}
}

// ShiftAllLeft shifts each element of x left by shift bits.
// If shift is greater than the element width, the result is 0.
//
//	z[i] = x[i] << shift
func (x Int32s) ShiftAllLeft(shift uint64) (z Int32s) {
	var res Int32s
	for i := 0; i < 4; i++ {
		res.set(i, x.get(i)<<shift)
	}
	return res
}

// ShiftAllRight arithmetically shifts each element of x right by shift bits.
// If shift is greater than the element width, the result is 0 or -1.
//
//	z[i] = x[i] >> shift
func (x Int32s) ShiftAllRight(shift uint64) (z Int32s) {
	var res Int32s
	for i := 0; i < 4; i++ {
		res.set(i, x.get(i)>>shift)
	}
	return res
}

// ReduceSum returns the scalar sum of the elements of x.
//
//	z = x[0] + x[1] + ...
func (x Int32s) ReduceSum() (z int32) {
	var res int32
	for i := 0; i < 4; i++ {
		res += x.get(i)
	}
	return res
}

// Store stores the elements of x into a slice. If len(s) < x.Len(), it panics.
func (x Int32s) Store(s []int32) {
	for i := 0; i < 4 && i < len(s); i++ {
		s[i] = x.get(i)
	}
}

// StorePart stores n=min(len(s), x.Len()) elements of x into s and returns n.
func (x Int32s) StorePart(s []int32) (n int) {
	x.Store(s)
	return min(len(s), x.Len())
}

// String returns a string representation of the vector.
func (x Int32s) String() string {
	var parts [4]int32
	for i := 0; i < 4; i++ {
		parts[i] = x.get(i)
	}
	return sliceToString(parts[:])
}

// Sub subtracts y from x elementwise.
//
//	z[i] = x[i] - y[i]
func (x Int32s) Sub(y Int32s) (z Int32s) {
	var res Int32s
	for i := 0; i < 4; i++ {
		res.set(i, x.get(i)-y.get(i))
	}
	return res
}

// ToMask returns a mask indicating which elements of x are non-zero.
//
//	z[i] = x[i] != 0
func (x Int32s) ToMask() (z Mask32s) {
	var res Mask32s
	for i := 0; i < 4; i++ {
		if x.get(i) != 0 {
			res.set(i, true)
		}
	}
	return res
}

// Xor returns the bitwise XOR of x and y.
//
//	z[i] = x[i] ^ y[i]
func (x Int32s) Xor(y Int32s) (z Int32s) {
	return Int32s{a: x.a ^ y.a, b: x.b ^ y.b}
}

// ConvertToUint32 converts each element of x to uint32.
func (x Int32s) ConvertToUint32() (z Uint32s) {
	return Uint32s{a: x.a, b: x.b}
}

// ToBits reinterprets the bits of each element of x as type uint32.
func (x Int32s) ToBits() (z Uint32s) {
	return Uint32s{a: x.a, b: x.b}
}

// LoadInt64s loads a slice into a vector. If len(s) < z.Len(), it panics.
func LoadInt64s(s []int64) (z Int64s) {
	var a, b uint64
	a = uint64(s[0])
	b = uint64(s[1])
	return Int64s{a: a, b: b}
}

// LoadInt64sPart loads n=min(len(s), z.Len()) elements from slice s as a
// vector and returns the vector and n. If len(s) < z.Len(), the
// remaining vector elements will be zero.
func LoadInt64sPart(s []int64) (z Int64s, n int) {
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

// Add adds x and y elementwise.
//
//	z[i] = x[i] + y[i]
func (x Int64s) Add(y Int64s) (z Int64s) {
	return Int64s{a: x.a + y.a, b: x.b + y.b}
}

// And returns the bitwise AND of x and y.
//
//	z[i] = x[i] & y[i]
func (x Int64s) And(y Int64s) (z Int64s) {
	return Int64s{a: x.a & y.a, b: x.b & y.b}
}

// AndNot returns the bitwise AND NOT of x and y.
//
//	z[i] = x[i] &^ y[i]
func (x Int64s) AndNot(y Int64s) (z Int64s) {
	return Int64s{a: x.a &^ y.a, b: x.b &^ y.b}
}

// Equal returns a mask indicating which elements of x and y are equal.
func (x Int64s) Equal(y Int64s) (z Mask64s) {
	var res Mask64s
	if x.a == y.a {
		res.a = ^uint64(0)
	}
	if x.b == y.b {
		res.b = ^uint64(0)
	}
	return res
}

// Greater returns a mask indicating which elements of x are greater than y.
//
//	z[i] = x[i] > y[i]
func (x Int64s) Greater(y Int64s) (z Mask64s) {
	var res Mask64s
	if int64(x.a) > int64(y.a) {
		res.a = ^uint64(0)
	}
	if int64(x.b) > int64(y.b) {
		res.b = ^uint64(0)
	}
	return res
}

// GreaterEqual returns a mask indicating which elements of x are greater than
// or equal to y.
//
//	z[i] = x[i] >= y[i]
func (x Int64s) GreaterEqual(y Int64s) (z Mask64s) {
	var res Mask64s
	if int64(x.a) >= int64(y.a) {
		res.a = ^uint64(0)
	}
	if int64(x.b) >= int64(y.b) {
		res.b = ^uint64(0)
	}
	return res
}

// Less returns a mask indicating which elements of x are less than y.
//
//	z[i] = x[i] < y[i]
func (x Int64s) Less(y Int64s) (z Mask64s) {
	var res Mask64s
	if int64(x.a) < int64(y.a) {
		res.a = ^uint64(0)
	}
	if int64(x.b) < int64(y.b) {
		res.b = ^uint64(0)
	}
	return res
}

// LessEqual returns a mask indicating which elements of x are less than or
// equal to y.
//
//	z[i] = x[i] <= y[i]
func (x Int64s) LessEqual(y Int64s) (z Mask64s) {
	var res Mask64s
	if int64(x.a) <= int64(y.a) {
		res.a = ^uint64(0)
	}
	if int64(x.b) <= int64(y.b) {
		res.b = ^uint64(0)
	}
	return res
}

// NotEqual returns a mask indicating which elements of x and y are not equal.
func (x Int64s) NotEqual(y Int64s) (z Mask64s) {
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

// Masked returns a vector with elements from x where mask is true, and zero
// elsewhere.
//
//	z[i] = if mask[i] { x[i] } else { 0 }
func (x Int64s) Masked(mask Mask64s) (z Int64s) {
	return Int64s{a: x.a & mask.a, b: x.b & mask.b}
}

// IfElse returns a vector with elements from x where mask is true, and y where
// mask is false.
//
//	z[i] = if mask[i] { x[i] } else { y[i] }
func (x Int64s) IfElse(mask Mask64s, y Int64s) (z Int64s) {
	return Int64s{
		a: (x.a & mask.a) | (y.a &^ mask.a),
		b: (x.b & mask.b) | (y.b &^ mask.b),
	}
}

// Neg returns the elementwise negation of x.
//
//	z[i] = -x[i]
func (x Int64s) Neg() (z Int64s) {
	return Int64s{a: uint64(-int64(x.a)), b: uint64(-int64(x.b))}
}

// Not returns the bitwise negation of x.
//
//	z[i] = ^x[i]
func (x Int64s) Not() (z Int64s) {
	return Int64s{a: ^x.a, b: ^x.b}
}

// Or returns the bitwise OR of x and y.
//
//	z[i] = x[i] | y[i]
func (x Int64s) Or(y Int64s) (z Int64s) {
	return Int64s{a: x.a | y.a, b: x.b | y.b}
}

// ShiftAllLeft shifts each element of x left by shift bits.
// If shift is greater than the element width, the result is 0.
//
//	z[i] = x[i] << shift
func (x Int64s) ShiftAllLeft(shift uint64) (z Int64s) {
	return Int64s{a: x.a << shift, b: x.b << shift}
}

// Store stores the elements of x into a slice. If len(s) < x.Len(), it panics.
func (x Int64s) Store(s []int64) {
	if len(s) > 0 {
		s[0] = int64(x.a)
	}
	if len(s) > 1 {
		s[1] = int64(x.b)
	}
}

// StorePart stores n=min(len(s), x.Len()) elements of x into s and returns n.
func (x Int64s) StorePart(s []int64) (n int) {
	x.Store(s)
	return min(len(s), x.Len())
}

// String returns a string representation of the vector.
func (x Int64s) String() string {
	return sliceToString([]int64{int64(x.a), int64(x.b)})
}

// Sub subtracts y from x elementwise.
//
//	z[i] = x[i] - y[i]
func (x Int64s) Sub(y Int64s) (z Int64s) {
	return Int64s{a: x.a - y.a, b: x.b - y.b}
}

// ToMask returns a mask indicating which elements of x are non-zero.
//
//	z[i] = x[i] != 0
func (x Int64s) ToMask() (z Mask64s) {
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
//
//	z[i] = x[i] ^ y[i]
func (x Int64s) Xor(y Int64s) (z Int64s) {
	return Int64s{a: x.a ^ y.a, b: x.b ^ y.b}
}

// ConvertToUint64 converts each element of x to uint64.
func (x Int64s) ConvertToUint64() (z Uint64s) {
	return Uint64s{a: x.a, b: x.b}
}

// ToBits reinterprets the bits of each element of x as type uint64.
func (x Int64s) ToBits() (z Uint64s) {
	return Uint64s{a: x.a, b: x.b}
}

// LoadUint8s loads a slice into a vector. If len(s) < z.Len(), it panics.
func LoadUint8s(s []uint8) (z Uint8s) {
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

// LoadUint8sPart loads n=min(len(s), z.Len()) elements from slice s as a
// vector and returns the vector and n. If len(s) < z.Len(), the
// remaining vector elements will be zero.
func LoadUint8sPart(s []uint8) (z Uint8s, n int) {
	var a, b uint64
	n = len(s)
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

// Add adds x and y elementwise.
//
//	z[i] = x[i] + y[i]
func (x Uint8s) Add(y Uint8s) (z Uint8s) {
	var res Uint8s
	for i := 0; i < 16; i++ {
		res.set(i, x.get(i)+y.get(i))
	}
	return res
}

// AddSaturated adds x and y elementwise with saturation.
//
//	z[i] = sat(x[i] + y[i])
func (x Uint8s) AddSaturated(y Uint8s) (z Uint8s) {
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
//
//	z[i] = x[i] & y[i]
func (x Uint8s) And(y Uint8s) (z Uint8s) {
	return Uint8s{a: x.a & y.a, b: x.b & y.b}
}

// AndNot returns the bitwise AND NOT of x and y.
//
//	z[i] = x[i] &^ y[i]
func (x Uint8s) AndNot(y Uint8s) (z Uint8s) {
	return Uint8s{a: x.a &^ y.a, b: x.b &^ y.b}
}

// Average returns the elementwise average of x and y, rounded toward +∞.
//
//	z[i] = (x[i] + y[i] + 1) / 2
func (x Uint8s) Average(y Uint8s) (z Uint8s) {
	var res Uint8s
	for i := 0; i < 16; i++ {
		res.set(i, uint8((int(x.get(i))+int(y.get(i))+1)>>1))
	}
	return res
}

// Equal returns a mask indicating which elements of x and y are equal.
func (x Uint8s) Equal(y Uint8s) (z Mask8s) {
	var res Mask8s
	for i := 0; i < 16; i++ {
		if x.get(i) == y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// NotEqual returns a mask indicating which elements of x and y are not equal.
func (x Uint8s) NotEqual(y Uint8s) (z Mask8s) {
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

// Masked returns a vector with elements from x where mask is true, and zero
// elsewhere.
//
//	z[i] = if mask[i] { x[i] } else { 0 }
func (x Uint8s) Masked(mask Mask8s) (z Uint8s) {
	return Uint8s{a: x.a & mask.a, b: x.b & mask.b}
}

// Max returns the elementwise maximum of x and y.
//
//	z[i] = max(x[i], y[i])
func (x Uint8s) Max(y Uint8s) (z Uint8s) {
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

// IfElse returns a vector with elements from x where mask is true, and y where
// mask is false.
//
//	z[i] = if mask[i] { x[i] } else { y[i] }
func (x Uint8s) IfElse(mask Mask8s, y Uint8s) (z Uint8s) {
	return Uint8s{
		a: (x.a & mask.a) | (y.a &^ mask.a),
		b: (x.b & mask.b) | (y.b &^ mask.b),
	}
}

// Min returns the elementwise minimum of x and y.
//
//	z[i] = min(x[i], y[i])
func (x Uint8s) Min(y Uint8s) (z Uint8s) {
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

// Mul returns the elementwise product of x and y.
//
//	z[i] = x[i] * y[i]
func (x Uint8s) Mul(y Uint8s) (z Uint8s) {
	var res Uint8s
	for i := 0; i < 16; i++ {
		res.set(i, x.get(i)*y.get(i))
	}
	return res
}

// Not returns the bitwise negation of x.
//
//	z[i] = ^x[i]
func (x Uint8s) Not() (z Uint8s) {
	return Uint8s{a: ^x.a, b: ^x.b}
}

// Or returns the bitwise OR of x and y.
//
//	z[i] = x[i] | y[i]
func (x Uint8s) Or(y Uint8s) (z Uint8s) {
	return Uint8s{a: x.a | y.a, b: x.b | y.b}
}

// ReduceSum returns the scalar sum of the elements of x.
//
//	z = x[0] + x[1] + ...
func (x Uint8s) ReduceSum() (z uint8) {
	var res uint8
	for i := 0; i < 16; i++ {
		res += x.get(i)
	}
	return res
}

// Store stores the elements of x into a slice. If len(s) < x.Len(), it panics.
func (x Uint8s) Store(s []uint8) {
	for i := 0; i < 16 && i < len(s); i++ {
		s[i] = x.get(i)
	}
}

// StorePart stores n=min(len(s), x.Len()) elements of x into s and returns n.
func (x Uint8s) StorePart(s []uint8) (n int) {
	x.Store(s)
	return min(len(s), x.Len())
}

// String returns a string representation of the vector.
func (x Uint8s) String() string {
	var parts [16]uint8
	for i := 0; i < 16; i++ {
		parts[i] = x.get(i)
	}
	return sliceToString(parts[:])
}

// Sub subtracts y from x elementwise.
//
//	z[i] = x[i] - y[i]
func (x Uint8s) Sub(y Uint8s) (z Uint8s) {
	var res Uint8s
	for i := 0; i < 16; i++ {
		res.set(i, x.get(i)-y.get(i))
	}
	return res
}

// SubSaturated subtracts y from x elementwise with saturation.
//
//	z[i] = sat(x[i] - y[i])
func (x Uint8s) SubSaturated(y Uint8s) (z Uint8s) {
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
//
//	z[i] = x[i] ^ y[i]
func (x Uint8s) Xor(y Uint8s) (z Uint8s) {
	return Uint8s{a: x.a ^ y.a, b: x.b ^ y.b}
}

// BitsToInt8 reinterprets the bits of each element of x as type int8.
func (x Uint8s) BitsToInt8() (z Int8s) {
	return Int8s{a: x.a, b: x.b}
}

// ConvertToInt8 converts each element of x to int8.
func (x Uint8s) ConvertToInt8() (z Int8s) {
	return Int8s{a: x.a, b: x.b}
}

// ReshapeToUint16s reinterprets the bits of x as a Uint16s vector.
//
// Both the vector elements and the bits of each element are interpreted in
// little endian order.
//
//	⋯     x[3]      x[2]      x[1]      x[0]
//	⋯ | 7  .. 0 | 7  .. 0 | 7  .. 0 | 7  .. 0 |
//	⋯ | 15     ....     0 | 15     ....     0 |
//	⋯          z[1]                z[0]
func (x Uint8s) ReshapeToUint16s() (z Uint16s) {
	return Uint16s{a: x.a, b: x.b}
}

// ReshapeToUint32s reinterprets the bits of x as a Uint32s vector.
//
// Both the vector elements and the bits of each element are interpreted in
// little endian order.
//
//	⋯     x[7]      x[6]      x[5]      x[4]      x[3]      x[2]      x[1]      x[0]
//	⋯ | 7  .. 0 | 7  .. 0 | 7  .. 0 | 7  .. 0 | 7  .. 0 | 7  .. 0 | 7  .. 0 | 7  .. 0 |
//	⋯ | 31               ....               0 | 31               ....               0 |
//	⋯                    z[1]                                    z[0]
func (x Uint8s) ReshapeToUint32s() (z Uint32s) {
	return Uint32s{a: x.a, b: x.b}
}

// ReshapeToUint64s reinterprets the bits of x as a Uint64s vector.
//
// Both the vector elements and the bits of each element are interpreted in
// little endian order.
//
//	⋯    x[15]     x[14]    ⋯     x[9]      x[8]      x[7]      x[6]    ⋯     x[1]      x[0]
//	⋯ | 7  .. 0 | 7  .. 0 | ⋯ | 7  .. 0 | 7  .. 0 | 7  .. 0 | 7  .. 0 | ⋯ | 7  .. 0 | 7  .. 0 |
//	⋯ | 63                 ....                 0 | 63                 ....                 0 |
//	⋯                      z[1]                                        z[0]
func (x Uint8s) ReshapeToUint64s() (z Uint64s) {
	return Uint64s{a: x.a, b: x.b}
}

// LoadUint16s loads a slice into a vector. If len(s) < z.Len(), it panics.
func LoadUint16s(s []uint16) (z Uint16s) {
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

// LoadUint16sPart loads n=min(len(s), z.Len()) elements from slice s as a
// vector and returns the vector and n. If len(s) < z.Len(), the
// remaining vector elements will be zero.
func LoadUint16sPart(s []uint16) (z Uint16s, n int) {
	var a, b uint64
	n = len(s)
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

// Add adds x and y elementwise.
//
//	z[i] = x[i] + y[i]
func (x Uint16s) Add(y Uint16s) (z Uint16s) {
	var res Uint16s
	for i := 0; i < 8; i++ {
		res.set(i, x.get(i)+y.get(i))
	}
	return res
}

// AddSaturated adds x and y elementwise with saturation.
//
//	z[i] = sat(x[i] + y[i])
func (x Uint16s) AddSaturated(y Uint16s) (z Uint16s) {
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
//
//	z[i] = x[i] & y[i]
func (x Uint16s) And(y Uint16s) (z Uint16s) {
	return Uint16s{a: x.a & y.a, b: x.b & y.b}
}

// AndNot returns the bitwise AND NOT of x and y.
//
//	z[i] = x[i] &^ y[i]
func (x Uint16s) AndNot(y Uint16s) (z Uint16s) {
	return Uint16s{a: x.a &^ y.a, b: x.b &^ y.b}
}

// Average returns the elementwise average of x and y, rounded toward +∞.
//
//	z[i] = (x[i] + y[i] + 1) / 2
func (x Uint16s) Average(y Uint16s) (z Uint16s) {
	var res Uint16s
	for i := 0; i < 8; i++ {
		res.set(i, uint16((int(x.get(i))+int(y.get(i))+1)>>1))
	}
	return res
}

// Equal returns a mask indicating which elements of x and y are equal.
func (x Uint16s) Equal(y Uint16s) (z Mask16s) {
	var res Mask16s
	for i := 0; i < 8; i++ {
		if x.get(i) == y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// Greater returns a mask indicating which elements of x are greater than y.
//
//	z[i] = x[i] > y[i]
func (x Uint16s) Greater(y Uint16s) (z Mask16s) {
	var res Mask16s
	for i := 0; i < 8; i++ {
		if x.get(i) > y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// GreaterEqual returns a mask indicating which elements of x are greater than
// or equal to y.
//
//	z[i] = x[i] >= y[i]
func (x Uint16s) GreaterEqual(y Uint16s) (z Mask16s) {
	var res Mask16s
	for i := 0; i < 8; i++ {
		if x.get(i) >= y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// Less returns a mask indicating which elements of x are less than y.
//
//	z[i] = x[i] < y[i]
func (x Uint16s) Less(y Uint16s) (z Mask16s) {
	var res Mask16s
	for i := 0; i < 8; i++ {
		if x.get(i) < y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// LessEqual returns a mask indicating which elements of x are less than or
// equal to y.
//
//	z[i] = x[i] <= y[i]
func (x Uint16s) LessEqual(y Uint16s) (z Mask16s) {
	var res Mask16s
	for i := 0; i < 8; i++ {
		if x.get(i) <= y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// NotEqual returns a mask indicating which elements of x and y are not equal.
func (x Uint16s) NotEqual(y Uint16s) (z Mask16s) {
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

// Masked returns a vector with elements from x where mask is true, and zero
// elsewhere.
//
//	z[i] = if mask[i] { x[i] } else { 0 }
func (x Uint16s) Masked(mask Mask16s) (z Uint16s) {
	return Uint16s{a: x.a & mask.a, b: x.b & mask.b}
}

// Max returns the elementwise maximum of x and y.
//
//	z[i] = max(x[i], y[i])
func (x Uint16s) Max(y Uint16s) (z Uint16s) {
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

// IfElse returns a vector with elements from x where mask is true, and y where
// mask is false.
//
//	z[i] = if mask[i] { x[i] } else { y[i] }
func (x Uint16s) IfElse(mask Mask16s, y Uint16s) (z Uint16s) {
	return Uint16s{
		a: (x.a & mask.a) | (y.a &^ mask.a),
		b: (x.b & mask.b) | (y.b &^ mask.b),
	}
}

// Min returns the elementwise minimum of x and y.
//
//	z[i] = min(x[i], y[i])
func (x Uint16s) Min(y Uint16s) (z Uint16s) {
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

// Mul returns the elementwise product of x and y.
//
//	z[i] = x[i] * y[i]
func (x Uint16s) Mul(y Uint16s) (z Uint16s) {
	var res Uint16s
	for i := 0; i < 8; i++ {
		res.set(i, x.get(i)*y.get(i))
	}
	return res
}

// Not returns the bitwise negation of x.
//
//	z[i] = ^x[i]
func (x Uint16s) Not() (z Uint16s) {
	return Uint16s{a: ^x.a, b: ^x.b}
}

// Or returns the bitwise OR of x and y.
//
//	z[i] = x[i] | y[i]
func (x Uint16s) Or(y Uint16s) (z Uint16s) {
	return Uint16s{a: x.a | y.a, b: x.b | y.b}
}

// ShiftAllLeft shifts each element of x left by shift bits.
// If shift is greater than the element width, the result is 0.
//
//	z[i] = x[i] << shift
func (x Uint16s) ShiftAllLeft(shift uint64) (z Uint16s) {
	var res Uint16s
	for i := 0; i < 8; i++ {
		res.set(i, x.get(i)<<shift)
	}
	return res
}

// ShiftAllRight logically shifts each element of x right by shift bits.
// If shift is greater than the element width, the result is 0.
//
//	z[i] = x[i] >> shift
func (x Uint16s) ShiftAllRight(shift uint64) (z Uint16s) {
	var res Uint16s
	for i := 0; i < 8; i++ {
		res.set(i, x.get(i)>>shift)
	}
	return res
}

// RotateAllLeft rotates each element of x left by shift bits.
//
//	z[i] = rotateLeft(x[i], shift)
func (x Uint16s) RotateAllLeft(shift uint64) (z Uint16s) {
	var res Uint16s
	d := shift & 15
	for i := 0; i < 8; i++ {
		u := x.get(i)
		r := (u << d) | (u >> ((16 - d) & 15))
		res.set(i, r)
	}
	return res
}

// RotateAllRight rotates each element of x right by shift bits.
//
//	z[i] = rotateRight(x[i], shift)
func (x Uint16s) RotateAllRight(shift uint64) (z Uint16s) {
	var res Uint16s
	d := shift & 15
	for i := 0; i < 8; i++ {
		u := x.get(i)
		r := (u >> d) | (u << ((16 - d) & 15))
		res.set(i, r)
	}
	return res
}

// ReduceSum returns the scalar sum of the elements of x.
//
//	z = x[0] + x[1] + ...
func (x Uint16s) ReduceSum() (z uint16) {
	var res uint16
	for i := 0; i < 8; i++ {
		res += x.get(i)
	}
	return res
}

// Store stores the elements of x into a slice. If len(s) < x.Len(), it panics.
func (x Uint16s) Store(s []uint16) {
	for i := 0; i < 8 && i < len(s); i++ {
		s[i] = x.get(i)
	}
}

// StorePart stores n=min(len(s), x.Len()) elements of x into s and returns n.
func (x Uint16s) StorePart(s []uint16) (n int) {
	x.Store(s)
	return min(len(s), x.Len())
}

// String returns a string representation of the vector.
func (x Uint16s) String() string {
	var parts [8]uint16
	for i := 0; i < 8; i++ {
		parts[i] = x.get(i)
	}
	return sliceToString(parts[:])
}

// Sub subtracts y from x elementwise.
//
//	z[i] = x[i] - y[i]
func (x Uint16s) Sub(y Uint16s) (z Uint16s) {
	var res Uint16s
	for i := 0; i < 8; i++ {
		res.set(i, x.get(i)-y.get(i))
	}
	return res
}

// SubSaturated subtracts y from x elementwise with saturation.
//
//	z[i] = sat(x[i] - y[i])
func (x Uint16s) SubSaturated(y Uint16s) (z Uint16s) {
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
//
//	z[i] = x[i] ^ y[i]
func (x Uint16s) Xor(y Uint16s) (z Uint16s) {
	return Uint16s{a: x.a ^ y.a, b: x.b ^ y.b}
}

// BitsToInt16 reinterprets the bits of each element of x as type int16.
func (x Uint16s) BitsToInt16() (z Int16s) {
	return Int16s{a: x.a, b: x.b}
}

// ConvertToInt16 converts each element of x to int16.
func (x Uint16s) ConvertToInt16() (z Int16s) {
	return Int16s{a: x.a, b: x.b}
}

// ReshapeToUint32s reinterprets the bits of x as a Uint32s vector.
//
// Both the vector elements and the bits of each element are interpreted in
// little endian order.
//
//	⋯     x[3]      x[2]      x[1]      x[0]
//	⋯ | 15 .. 0 | 15 .. 0 | 15 .. 0 | 15 .. 0 |
//	⋯ | 31     ....     0 | 31     ....     0 |
//	⋯          z[1]                z[0]
func (x Uint16s) ReshapeToUint32s() (z Uint32s) {
	return Uint32s{a: x.a, b: x.b}
}

// ReshapeToUint64s reinterprets the bits of x as a Uint64s vector.
//
// Both the vector elements and the bits of each element are interpreted in
// little endian order.
//
//	⋯     x[7]      x[6]      x[5]      x[4]      x[3]      x[2]      x[1]      x[0]
//	⋯ | 15 .. 0 | 15 .. 0 | 15 .. 0 | 15 .. 0 | 15 .. 0 | 15 .. 0 | 15 .. 0 | 15 .. 0 |
//	⋯ | 63               ....               0 | 63               ....               0 |
//	⋯                    z[1]                                    z[0]
func (x Uint16s) ReshapeToUint64s() (z Uint64s) {
	return Uint64s{a: x.a, b: x.b}
}

// ReshapeToUint8s reinterprets the bits of x as a Uint8s vector.
//
// Both the vector elements and the bits of each element are interpreted in
// little endian order.
//
//	⋯          x[1]                x[0]
//	⋯ | 15     ....     0 | 15     ....     0 |
//	⋯ | 7  .. 0 | 7  .. 0 | 7  .. 0 | 7  .. 0 |
//	⋯     z[3]      z[2]      z[1]      z[0]
func (x Uint16s) ReshapeToUint8s() (z Uint8s) {
	return Uint8s{a: x.a, b: x.b}
}

// LoadUint32s loads a slice into a vector. If len(s) < z.Len(), it panics.
func LoadUint32s(s []uint32) (z Uint32s) {
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

// LoadUint32sPart loads n=min(len(s), z.Len()) elements from slice s as a
// vector and returns the vector and n. If len(s) < z.Len(), the
// remaining vector elements will be zero.
func LoadUint32sPart(s []uint32) (z Uint32s, n int) {
	var a, b uint64
	n = len(s)
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
		mask := uint64(0xffff_ffff) << (32 * i)
		x.a = (x.a &^ mask) | (val << (32 * i))
	} else {
		mask := uint64(0xffff_ffff) << (32 * (i - 2))
		x.b = (x.b &^ mask) | (val << (32 * (i - 2)))
	}
}

// Add adds x and y elementwise.
//
//	z[i] = x[i] + y[i]
func (x Uint32s) Add(y Uint32s) (z Uint32s) {
	var res Uint32s
	for i := 0; i < 4; i++ {
		res.set(i, x.get(i)+y.get(i))
	}
	return res
}

// And returns the bitwise AND of x and y.
//
//	z[i] = x[i] & y[i]
func (x Uint32s) And(y Uint32s) (z Uint32s) {
	return Uint32s{a: x.a & y.a, b: x.b & y.b}
}

// AndNot returns the bitwise AND NOT of x and y.
//
//	z[i] = x[i] &^ y[i]
func (x Uint32s) AndNot(y Uint32s) (z Uint32s) {
	return Uint32s{a: x.a &^ y.a, b: x.b &^ y.b}
}

// Equal returns a mask indicating which elements of x and y are equal.
func (x Uint32s) Equal(y Uint32s) (z Mask32s) {
	var res Mask32s
	for i := 0; i < 4; i++ {
		if x.get(i) == y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// Greater returns a mask indicating which elements of x are greater than y.
//
//	z[i] = x[i] > y[i]
func (x Uint32s) Greater(y Uint32s) (z Mask32s) {
	var res Mask32s
	for i := 0; i < 4; i++ {
		if x.get(i) > y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// GreaterEqual returns a mask indicating which elements of x are greater than
// or equal to y.
//
//	z[i] = x[i] >= y[i]
func (x Uint32s) GreaterEqual(y Uint32s) (z Mask32s) {
	var res Mask32s
	for i := 0; i < 4; i++ {
		if x.get(i) >= y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// Less returns a mask indicating which elements of x are less than y.
//
//	z[i] = x[i] < y[i]
func (x Uint32s) Less(y Uint32s) (z Mask32s) {
	var res Mask32s
	for i := 0; i < 4; i++ {
		if x.get(i) < y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// LessEqual returns a mask indicating which elements of x are less than or
// equal to y.
//
//	z[i] = x[i] <= y[i]
func (x Uint32s) LessEqual(y Uint32s) (z Mask32s) {
	var res Mask32s
	for i := 0; i < 4; i++ {
		if x.get(i) <= y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// NotEqual returns a mask indicating which elements of x and y are not equal.
func (x Uint32s) NotEqual(y Uint32s) (z Mask32s) {
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

// Masked returns a vector with elements from x where mask is true, and zero
// elsewhere.
//
//	z[i] = if mask[i] { x[i] } else { 0 }
func (x Uint32s) Masked(mask Mask32s) (z Uint32s) {
	return Uint32s{a: x.a & mask.a, b: x.b & mask.b}
}

// Max returns the elementwise maximum of x and y.
//
//	z[i] = max(x[i], y[i])
func (x Uint32s) Max(y Uint32s) (z Uint32s) {
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

// IfElse returns a vector with elements from x where mask is true, and y where
// mask is false.
//
//	z[i] = if mask[i] { x[i] } else { y[i] }
func (x Uint32s) IfElse(mask Mask32s, y Uint32s) (z Uint32s) {
	return Uint32s{
		a: (x.a & mask.a) | (y.a &^ mask.a),
		b: (x.b & mask.b) | (y.b &^ mask.b),
	}
}

// Min returns the elementwise minimum of x and y.
//
//	z[i] = min(x[i], y[i])
func (x Uint32s) Min(y Uint32s) (z Uint32s) {
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

// Mul returns the elementwise product of x and y.
//
//	z[i] = x[i] * y[i]
func (x Uint32s) Mul(y Uint32s) (z Uint32s) {
	var res Uint32s
	for i := 0; i < 4; i++ {
		res.set(i, x.get(i)*y.get(i))
	}
	return res
}

// Not returns the bitwise negation of x.
//
//	z[i] = ^x[i]
func (x Uint32s) Not() (z Uint32s) {
	return Uint32s{a: ^x.a, b: ^x.b}
}

// Or returns the bitwise OR of x and y.
//
//	z[i] = x[i] | y[i]
func (x Uint32s) Or(y Uint32s) (z Uint32s) {
	return Uint32s{a: x.a | y.a, b: x.b | y.b}
}

// ShiftAllLeft shifts each element of x left by shift bits.
// If shift is greater than the element width, the result is 0.
//
//	z[i] = x[i] << shift
func (x Uint32s) ShiftAllLeft(shift uint64) (z Uint32s) {
	var res Uint32s
	for i := 0; i < 4; i++ {
		res.set(i, x.get(i)<<shift)
	}
	return res
}

// ShiftAllRight logically shifts each element of x right by shift bits.
// If shift is greater than the element width, the result is 0.
//
//	z[i] = x[i] >> shift
func (x Uint32s) ShiftAllRight(shift uint64) (z Uint32s) {
	var res Uint32s
	for i := 0; i < 4; i++ {
		res.set(i, x.get(i)>>shift)
	}
	return res
}

// RotateAllLeft rotates each element of x left by shift bits.
//
//	z[i] = rotateLeft(x[i], shift)
func (x Uint32s) RotateAllLeft(shift uint64) (z Uint32s) {
	var res Uint32s
	d := shift & 31
	for i := 0; i < 4; i++ {
		u := x.get(i)
		r := (u << d) | (u >> ((32 - d) & 31))
		res.set(i, r)
	}
	return res
}

// RotateAllRight rotates each element of x right by shift bits.
//
//	z[i] = rotateRight(x[i], shift)
func (x Uint32s) RotateAllRight(shift uint64) (z Uint32s) {
	var res Uint32s
	d := shift & 31
	for i := 0; i < 4; i++ {
		u := x.get(i)
		r := (u >> d) | (u << ((32 - d) & 31))
		res.set(i, r)
	}
	return res
}

// ReduceSum returns the scalar sum of the elements of x.
//
//	z = x[0] + x[1] + ...
func (x Uint32s) ReduceSum() (z uint32) {
	var res uint32
	for i := 0; i < 4; i++ {
		res += x.get(i)
	}
	return res
}

// Store stores the elements of x into a slice. If len(s) < x.Len(), it panics.
func (x Uint32s) Store(s []uint32) {
	for i := 0; i < 4 && i < len(s); i++ {
		s[i] = x.get(i)
	}
}

// StorePart stores n=min(len(s), x.Len()) elements of x into s and returns n.
func (x Uint32s) StorePart(s []uint32) (n int) {
	x.Store(s)
	return min(len(s), x.Len())
}

// String returns a string representation of the vector.
func (x Uint32s) String() string {
	var parts [4]uint32
	for i := 0; i < 4; i++ {
		parts[i] = x.get(i)
	}
	return sliceToString(parts[:])
}

// Sub subtracts y from x elementwise.
//
//	z[i] = x[i] - y[i]
func (x Uint32s) Sub(y Uint32s) (z Uint32s) {
	var res Uint32s
	for i := 0; i < 4; i++ {
		res.set(i, x.get(i)-y.get(i))
	}
	return res
}

// Xor returns the bitwise XOR of x and y.
//
//	z[i] = x[i] ^ y[i]
func (x Uint32s) Xor(y Uint32s) (z Uint32s) {
	return Uint32s{a: x.a ^ y.a, b: x.b ^ y.b}
}

// BitsToFloat32 reinterprets the bits of each element of x as type float32.
func (x Uint32s) BitsToFloat32() (z Float32s) {
	return Float32s{a: x.a, b: x.b}
}

// BitsToInt32 reinterprets the bits of each element of x as type int32.
func (x Uint32s) BitsToInt32() (z Int32s) {
	return Int32s{a: x.a, b: x.b}
}

// ConvertToInt32 converts each element of x to int32.
func (x Uint32s) ConvertToInt32() (z Int32s) {
	return Int32s{a: x.a, b: x.b}
}

// ReshapeToUint16s reinterprets the bits of x as a Uint16s vector.
//
// Both the vector elements and the bits of each element are interpreted in
// little endian order.
//
//	⋯          x[1]                x[0]
//	⋯ | 31     ....     0 | 31     ....     0 |
//	⋯ | 15 .. 0 | 15 .. 0 | 15 .. 0 | 15 .. 0 |
//	⋯     z[3]      z[2]      z[1]      z[0]
func (x Uint32s) ReshapeToUint16s() (z Uint16s) {
	return Uint16s{a: x.a, b: x.b}
}

// ReshapeToUint64s reinterprets the bits of x as a Uint64s vector.
//
// Both the vector elements and the bits of each element are interpreted in
// little endian order.
//
//	⋯     x[3]      x[2]      x[1]      x[0]
//	⋯ | 31 .. 0 | 31 .. 0 | 31 .. 0 | 31 .. 0 |
//	⋯ | 63     ....     0 | 63     ....     0 |
//	⋯          z[1]                z[0]
func (x Uint32s) ReshapeToUint64s() (z Uint64s) {
	return Uint64s{a: x.a, b: x.b}
}

// ReshapeToUint8s reinterprets the bits of x as a Uint8s vector.
//
// Both the vector elements and the bits of each element are interpreted in
// little endian order.
//
//	⋯                    x[1]                                    x[0]
//	⋯ | 31               ....               0 | 31               ....               0 |
//	⋯ | 7  .. 0 | 7  .. 0 | 7  .. 0 | 7  .. 0 | 7  .. 0 | 7  .. 0 | 7  .. 0 | 7  .. 0 |
//	⋯     z[7]      z[6]      z[5]      z[4]      z[3]      z[2]      z[1]      z[0]
func (x Uint32s) ReshapeToUint8s() (z Uint8s) {
	return Uint8s{a: x.a, b: x.b}
}

// LoadUint64s loads a slice into a vector. If len(s) < z.Len(), it panics.
func LoadUint64s(s []uint64) (z Uint64s) {
	var a, b uint64
	a = s[0]
	b = s[1]
	return Uint64s{a: a, b: b}
}

// LoadUint64sPart loads n=min(len(s), z.Len()) elements from slice s as a
// vector and returns the vector and n. If len(s) < z.Len(), the
// remaining vector elements will be zero.
func LoadUint64sPart(s []uint64) (z Uint64s, n int) {
	n = len(s)
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

// Add adds x and y elementwise.
//
//	z[i] = x[i] + y[i]
func (x Uint64s) Add(y Uint64s) (z Uint64s) {
	return Uint64s{a: x.a + y.a, b: x.b + y.b}
}

// And returns the bitwise AND of x and y.
//
//	z[i] = x[i] & y[i]
func (x Uint64s) And(y Uint64s) (z Uint64s) {
	return Uint64s{a: x.a & y.a, b: x.b & y.b}
}

// AndNot returns the bitwise AND NOT of x and y.
//
//	z[i] = x[i] &^ y[i]
func (x Uint64s) AndNot(y Uint64s) (z Uint64s) {
	return Uint64s{a: x.a &^ y.a, b: x.b &^ y.b}
}

// Equal returns a mask indicating which elements of x and y are equal.
func (x Uint64s) Equal(y Uint64s) (z Mask64s) {
	var res Mask64s
	if x.a == y.a {
		res.a = ^uint64(0)
	}
	if x.b == y.b {
		res.b = ^uint64(0)
	}
	return res
}

// Greater returns a mask indicating which elements of x are greater than y.
//
//	z[i] = x[i] > y[i]
func (x Uint64s) Greater(y Uint64s) (z Mask64s) {
	var res Mask64s
	for i := 0; i < 2; i++ {
		if x.get(i) > y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// GreaterEqual returns a mask indicating which elements of x are greater than
// or equal to y.
//
//	z[i] = x[i] >= y[i]
func (x Uint64s) GreaterEqual(y Uint64s) (z Mask64s) {
	var res Mask64s
	for i := 0; i < 2; i++ {
		if x.get(i) >= y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// Less returns a mask indicating which elements of x are less than y.
//
//	z[i] = x[i] < y[i]
func (x Uint64s) Less(y Uint64s) (z Mask64s) {
	var res Mask64s
	for i := 0; i < 2; i++ {
		if x.get(i) < y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// LessEqual returns a mask indicating which elements of x are less than or
// equal to y.
//
//	z[i] = x[i] <= y[i]
func (x Uint64s) LessEqual(y Uint64s) (z Mask64s) {
	var res Mask64s
	for i := 0; i < 2; i++ {
		if x.get(i) <= y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// NotEqual returns a mask indicating which elements of x and y are not equal.
func (x Uint64s) NotEqual(y Uint64s) (z Mask64s) {
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

// Masked returns a vector with elements from x where mask is true, and zero
// elsewhere.
//
//	z[i] = if mask[i] { x[i] } else { 0 }
func (x Uint64s) Masked(mask Mask64s) (z Uint64s) {
	return Uint64s{a: x.a & mask.a, b: x.b & mask.b}
}

// IfElse returns a vector with elements from x where mask is true, and y where
// mask is false.
//
//	z[i] = if mask[i] { x[i] } else { y[i] }
func (x Uint64s) IfElse(mask Mask64s, y Uint64s) (z Uint64s) {
	return Uint64s{
		a: (x.a & mask.a) | (y.a &^ mask.a),
		b: (x.b & mask.b) | (y.b &^ mask.b),
	}
}

// Not returns the bitwise negation of x.
//
//	z[i] = ^x[i]
func (x Uint64s) Not() (z Uint64s) {
	return Uint64s{a: ^x.a, b: ^x.b}
}

// Or returns the bitwise OR of x and y.
//
//	z[i] = x[i] | y[i]
func (x Uint64s) Or(y Uint64s) (z Uint64s) {
	return Uint64s{a: x.a | y.a, b: x.b | y.b}
}

// ShiftAllLeft shifts each element of x left by shift bits.
// If shift is greater than the element width, the result is 0.
//
//	z[i] = x[i] << shift
func (x Uint64s) ShiftAllLeft(shift uint64) (z Uint64s) {
	return Uint64s{a: x.a << shift, b: x.b << shift}
}

// ShiftAllRight logically shifts each element of x right by shift bits.
// If shift is greater than the element width, the result is 0.
//
//	z[i] = x[i] >> shift
func (x Uint64s) ShiftAllRight(shift uint64) (z Uint64s) {
	return Uint64s{a: x.a >> shift, b: x.b >> shift}
}

// RotateAllLeft rotates each element of x left by shift bits.
//
//	z[i] = rotateLeft(x[i], shift)
func (x Uint64s) RotateAllLeft(shift uint64) (z Uint64s) {
	d := shift & 63
	return Uint64s{
		a: (x.a << d) | (x.a >> ((64 - d) & 63)),
		b: (x.b << d) | (x.b >> ((64 - d) & 63)),
	}
}

// RotateAllRight rotates each element of x right by shift bits.
//
//	z[i] = rotateRight(x[i], shift)
func (x Uint64s) RotateAllRight(shift uint64) (z Uint64s) {
	d := shift & 63
	return Uint64s{
		a: (x.a >> d) | (x.a << ((64 - d) & 63)),
		b: (x.b >> d) | (x.b << ((64 - d) & 63)),
	}
}

// Store stores the elements of x into a slice. If len(s) < x.Len(), it panics.
func (x Uint64s) Store(s []uint64) {
	if len(s) > 0 {
		s[0] = x.a
	}
	if len(s) > 1 {
		s[1] = x.b
	}
}

// StorePart stores n=min(len(s), x.Len()) elements of x into s and returns n.
func (x Uint64s) StorePart(s []uint64) (n int) {
	x.Store(s)
	return min(len(s), x.Len())
}

// String returns a string representation of the vector.
func (x Uint64s) String() string {
	return sliceToString([]uint64{x.a, x.b})
}

// Sub subtracts y from x elementwise.
//
//	z[i] = x[i] - y[i]
func (x Uint64s) Sub(y Uint64s) (z Uint64s) {
	return Uint64s{a: x.a - y.a, b: x.b - y.b}
}

// Xor returns the bitwise XOR of x and y.
//
//	z[i] = x[i] ^ y[i]
func (x Uint64s) Xor(y Uint64s) (z Uint64s) {
	return Uint64s{a: x.a ^ y.a, b: x.b ^ y.b}
}

// BitsToFloat64 reinterprets the bits of each element of x as type float64.
func (x Uint64s) BitsToFloat64() (z Float64s) {
	return Float64s{a: x.a, b: x.b}
}

// BitsToInt64 reinterprets the bits of each element of x as type int64.
func (x Uint64s) BitsToInt64() (z Int64s) {
	return Int64s{a: x.a, b: x.b}
}

// ConvertToInt64 converts each element of x to int64.
func (x Uint64s) ConvertToInt64() (z Int64s) {
	return Int64s{a: x.a, b: x.b}
}

// ReshapeToUint16s reinterprets the bits of x as a Uint16s vector.
//
// Both the vector elements and the bits of each element are interpreted in
// little endian order.
//
//	⋯                    x[1]                                    x[0]
//	⋯ | 63               ....               0 | 63               ....               0 |
//	⋯ | 15 .. 0 | 15 .. 0 | 15 .. 0 | 15 .. 0 | 15 .. 0 | 15 .. 0 | 15 .. 0 | 15 .. 0 |
//	⋯     z[7]      z[6]      z[5]      z[4]      z[3]      z[2]      z[1]      z[0]
func (x Uint64s) ReshapeToUint16s() (z Uint16s) {
	return Uint16s{a: x.a, b: x.b}
}

// ReshapeToUint32s reinterprets the bits of x as a Uint32s vector.
//
// Both the vector elements and the bits of each element are interpreted in
// little endian order.
//
//	⋯          x[1]                x[0]
//	⋯ | 63     ....     0 | 63     ....     0 |
//	⋯ | 31 .. 0 | 31 .. 0 | 31 .. 0 | 31 .. 0 |
//	⋯     z[3]      z[2]      z[1]      z[0]
func (x Uint64s) ReshapeToUint32s() (z Uint32s) {
	return Uint32s{a: x.a, b: x.b}
}

// ReshapeToUint8s reinterprets the bits of x as a Uint8s vector.
//
// Both the vector elements and the bits of each element are interpreted in
// little endian order.
//
//	⋯                      x[1]                                        x[0]
//	⋯ | 63                 ....                 0 | 63                 ....                 0 |
//	⋯ | 7  .. 0 | 7  .. 0 | ⋯ | 7  .. 0 | 7  .. 0 | 7  .. 0 | 7  .. 0 | ⋯ | 7  .. 0 | 7  .. 0 |
//	⋯    z[15]     z[14]    ⋯     z[9]      z[8]      z[7]      z[6]    ⋯     z[1]      z[0]
func (x Uint64s) ReshapeToUint8s() (z Uint8s) {
	return Uint8s{a: x.a, b: x.b}
}

// LoadFloat32s loads a slice into a vector. If len(s) < z.Len(), it panics.
func LoadFloat32s(s []float32) (z Float32s) {
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

// LoadFloat32sPart loads n=min(len(s), z.Len()) elements from slice s as a
// vector and returns the vector and n. If len(s) < z.Len(), the
// remaining vector elements will be zero.
func LoadFloat32sPart(s []float32) (z Float32s, n int) {
	var a, b uint64
	n = len(s)
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
		mask := uint64(0xffff_ffff) << (32 * i)
		x.a = (x.a &^ mask) | (val << (32 * i))
	} else {
		mask := uint64(0xffff_ffff) << (32 * (i - 2))
		x.b = (x.b &^ mask) | (val << (32 * (i - 2)))
	}
}

// Abs returns the elementwise absolute value of x.
func (x Float32s) Abs() (z Float32s) {
	var res Float32s
	for i := 0; i < 4; i++ {
		v := x.get(i)
		v = float32(math.Abs(float64(v)))
		res.set(i, v)
	}
	return res
}

// Add adds x and y elementwise.
//
//	z[i] = x[i] + y[i]
func (x Float32s) Add(y Float32s) (z Float32s) {
	var res Float32s
	res.set(0, x.get(0)+y.get(0))
	res.set(1, x.get(1)+y.get(1))
	res.set(2, x.get(2)+y.get(2))
	res.set(3, x.get(3)+y.get(3))
	return res
}

// ConvertToInt32 converts each element of x to int32.
//
// When a conversion is inexact, this truncates the result (rounds toward zero).
// If the converted result would be outside the representable range, the result
// is architecture-dependent.
func (x Float32s) ConvertToInt32() (z Int32s) {
	var res Int32s
	for i := 0; i < 4; i++ {
		res.set(i, int32(x.get(i)))
	}
	return res
}

// Div divides x by y elementwise.
//
//	z[i] = x[i] / y[i]
//
// Division by zero does not panic, and the result follows IEEE 754. That is,
// dividing a non-zero value by zero results in +/- infinity, and dividing zero
// by zero results in NaN.
func (x Float32s) Div(y Float32s) (z Float32s) {
	var res Float32s
	for i := 0; i < 4; i++ {
		res.set(i, x.get(i)/y.get(i))
	}
	return res
}

// Equal returns a mask indicating which elements of x and y are equal.
func (x Float32s) Equal(y Float32s) (z Mask32s) {
	var res Mask32s
	for i := 0; i < 4; i++ {
		if x.get(i) == y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// Greater returns a mask indicating which elements of x are greater than y.
//
//	z[i] = x[i] > y[i]
func (x Float32s) Greater(y Float32s) (z Mask32s) {
	var res Mask32s
	for i := 0; i < 4; i++ {
		if x.get(i) > y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// GreaterEqual returns a mask indicating which elements of x are greater than
// or equal to y.
//
//	z[i] = x[i] >= y[i]
func (x Float32s) GreaterEqual(y Float32s) (z Mask32s) {
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

// Less returns a mask indicating which elements of x are less than y.
//
//	z[i] = x[i] < y[i]
func (x Float32s) Less(y Float32s) (z Mask32s) {
	var res Mask32s
	for i := 0; i < 4; i++ {
		if x.get(i) < y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// LessEqual returns a mask indicating which elements of x are less than or
// equal to y.
//
//	z[i] = x[i] <= y[i]
func (x Float32s) LessEqual(y Float32s) (z Mask32s) {
	var res Mask32s
	for i := 0; i < 4; i++ {
		if x.get(i) <= y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// Masked returns a vector with elements from x where mask is true, and zero
// elsewhere.
//
//	z[i] = if mask[i] { x[i] } else { 0 }
func (x Float32s) Masked(mask Mask32s) (z Float32s) {
	return Float32s{a: x.a & mask.a, b: x.b & mask.b}
}

// Max returns the elementwise maximum of x and y.
//
//	z[i] = max(x[i], y[i])
func (x Float32s) Max(y Float32s) (z Float32s) {
	var res Float32s
	for i := 0; i < 4; i++ {
		vx := x.get(i)
		vy := y.get(i)
		res.set(i, max(vx, vy))
	}
	return res
}

// IfElse returns a vector with elements from x where mask is true, and y where
// mask is false.
//
//	z[i] = if mask[i] { x[i] } else { y[i] }
func (x Float32s) IfElse(mask Mask32s, y Float32s) (z Float32s) {
	return Float32s{
		a: (x.a & mask.a) | (y.a &^ mask.a),
		b: (x.b & mask.b) | (y.b &^ mask.b),
	}
}

// Min returns the elementwise minimum of x and y.
//
//	z[i] = min(x[i], y[i])
func (x Float32s) Min(y Float32s) (z Float32s) {
	var res Float32s
	for i := 0; i < 4; i++ {
		vx := x.get(i)
		vy := y.get(i)
		res.set(i, min(vx, vy))
	}
	return res
}

// Mul returns the elementwise product of x and y.
//
//	z[i] = x[i] * y[i]
func (x Float32s) Mul(y Float32s) (z Float32s) {
	var res Float32s
	res.set(0, x.get(0)*y.get(0))
	res.set(1, x.get(1)*y.get(1))
	res.set(2, x.get(2)*y.get(2))
	res.set(3, x.get(3)*y.get(3))
	return res
}

// MulAdd returns x * y + z elementwise.
//
//	w[i] = x[i] * y[i] + z[i]
func (x Float32s) MulAdd(y Float32s, z Float32s) (w Float32s) {
	var res Float32s

	res.set(0, x.get(0)*y.get(0)+z.get(0))
	res.set(1, x.get(1)*y.get(1)+z.get(1))
	res.set(2, x.get(2)*y.get(2)+z.get(2))
	res.set(3, x.get(3)*y.get(3)+z.get(3))
	return res
}

// Neg returns the elementwise negation of x.
//
//	z[i] = -x[i]
func (x Float32s) Neg() (z Float32s) {
	var res Float32s
	for i := 0; i < 4; i++ {
		res.set(i, -(x.get(i)))
	}
	return res
}

// NotEqual returns a mask indicating which elements of x and y are not equal.
func (x Float32s) NotEqual(y Float32s) (z Mask32s) {
	var res Mask32s
	for i := 0; i < 4; i++ {
		if x.get(i) != y.get(i) {
			res.set(i, true)
		}
	}
	return res
}

// ReduceSum returns the scalar sum of the elements of x.
//
//	z = x[0] + x[1] + ...
func (x Float32s) ReduceSum() (z float32) {
	// Evaluate with same associativity as the horizontal-add idiom.
	// It's also perhaps faster, since a shorter expression tree.
	return (x.get(0) + x.get(1)) + (x.get(2) + x.get(3))
}

// Sqrt returns the elementwise square root of x.
//
//	z[i] = sqrt(x[i])
func (x Float32s) Sqrt() (z Float32s) {
	var res Float32s
	for i := 0; i < 4; i++ {
		res.set(i, float32(math.Sqrt(float64(x.get(i)))))
	}
	return res
}

// Store stores the elements of x into a slice. If len(s) < x.Len(), it panics.
func (x Float32s) Store(s []float32) {
	for i := 0; i < 4 && i < len(s); i++ {
		s[i] = x.get(i)
	}
}

// StorePart stores n=min(len(s), x.Len()) elements of x into s and returns n.
func (x Float32s) StorePart(s []float32) (n int) {
	x.Store(s)
	return min(len(s), x.Len())
}

// String returns a string representation of the vector.
func (x Float32s) String() string {
	var parts [4]float32
	for i := 0; i < 4; i++ {
		parts[i] = x.get(i)
	}
	return sliceToString(parts[:])
}

// Sub subtracts y from x elementwise.
//
//	z[i] = x[i] - y[i]
func (x Float32s) Sub(y Float32s) (z Float32s) {
	var res Float32s
	for i := 0; i < 4; i++ {
		res.set(i, x.get(i)-y.get(i))
	}
	return res
}

// ToBits returns the IEEE 754 binary representation of each element of x.
func (x Float32s) ToBits() (z Uint32s) {
	return Uint32s{a: x.a, b: x.b}
}

// LoadFloat64s loads a slice into a vector. If len(s) < z.Len(), it panics.
func LoadFloat64s(s []float64) (z Float64s) {
	var a, b uint64
	a = math.Float64bits(s[0])
	b = math.Float64bits(s[1])
	return Float64s{a: a, b: b}
}

// LoadFloat64sPart loads n=min(len(s), z.Len()) elements from slice s as a
// vector and returns the vector and n. If len(s) < z.Len(), the
// remaining vector elements will be zero.
func LoadFloat64sPart(s []float64) (z Float64s, n int) {
	n = len(s)
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

// Abs returns the elementwise absolute value of x.
func (x Float64s) Abs() (z Float64s) {
	var res Float64s
	for i := 0; i < 4; i++ {
		v := x.get(i)
		v = math.Abs(v)
		res.set(i, v)
	}
	return res
}

// Add adds x and y elementwise.
//
//	z[i] = x[i] + y[i]
func (x Float64s) Add(y Float64s) (z Float64s) {
	var res Float64s
	res.set(0, x.get(0)+y.get(0))
	res.set(1, x.get(1)+y.get(1))
	return res
}

// Div divides x by y elementwise.
//
//	z[i] = x[i] / y[i]
//
// Division by zero does not panic, and the result follows IEEE 754. That is,
// dividing a non-zero value by zero results in +/- infinity, and dividing zero
// by zero results in NaN.
func (x Float64s) Div(y Float64s) (z Float64s) {
	var res Float64s
	res.set(0, x.get(0)/y.get(0))
	res.set(1, x.get(1)/y.get(1))
	return res
}

// Equal returns a mask indicating which elements of x and y are equal.
func (x Float64s) Equal(y Float64s) (z Mask64s) {
	var res Mask64s
	if x.get(0) == y.get(0) {
		res.a = ^uint64(0)
	}
	if x.get(1) == y.get(1) {
		res.b = ^uint64(0)
	}
	return res
}

// Greater returns a mask indicating which elements of x are greater than y.
//
//	z[i] = x[i] > y[i]
func (x Float64s) Greater(y Float64s) (z Mask64s) {
	var res Mask64s
	if x.get(0) > y.get(0) {
		res.a = ^uint64(0)
	}
	if x.get(1) > y.get(1) {
		res.b = ^uint64(0)
	}
	return res
}

// GreaterEqual returns a mask indicating which elements of x are greater than
// or equal to y.
//
//	z[i] = x[i] >= y[i]
func (x Float64s) GreaterEqual(y Float64s) (z Mask64s) {
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

// Less returns a mask indicating which elements of x are less than y.
//
//	z[i] = x[i] < y[i]
func (x Float64s) Less(y Float64s) (z Mask64s) {
	var res Mask64s
	if x.get(0) < y.get(0) {
		res.a = ^uint64(0)
	}
	if x.get(1) < y.get(1) {
		res.b = ^uint64(0)
	}
	return res
}

// LessEqual returns a mask indicating which elements of x are less than or
// equal to y.
//
//	z[i] = x[i] <= y[i]
func (x Float64s) LessEqual(y Float64s) (z Mask64s) {
	var res Mask64s
	if x.get(0) <= y.get(0) {
		res.a = ^uint64(0)
	}
	if x.get(1) <= y.get(1) {
		res.b = ^uint64(0)
	}
	return res
}

// Masked returns a vector with elements from x where mask is true, and zero
// elsewhere.
//
//	z[i] = if mask[i] { x[i] } else { 0 }
func (x Float64s) Masked(mask Mask64s) (z Float64s) {
	return Float64s{a: x.a & mask.a, b: x.b & mask.b}
}

// Max returns the elementwise maximum of x and y.
//
//	z[i] = max(x[i], y[i])
func (x Float64s) Max(y Float64s) (z Float64s) {
	var res Float64s
	vx := x.get(0)
	vy := y.get(0)
	res.set(0, max(vx, vy))
	vx = x.get(1)
	vy = y.get(1)
	res.set(1, max(vx, vy))
	return res
}

// IfElse returns a vector with elements from x where mask is true, and y where
// mask is false.
//
//	z[i] = if mask[i] { x[i] } else { y[i] }
func (x Float64s) IfElse(mask Mask64s, y Float64s) (z Float64s) {
	return Float64s{
		a: (x.a & mask.a) | (y.a &^ mask.a),
		b: (x.b & mask.b) | (y.b &^ mask.b),
	}
}

// Min returns the elementwise minimum of x and y.
//
//	z[i] = min(x[i], y[i])
func (x Float64s) Min(y Float64s) (z Float64s) {
	var res Float64s
	vx := x.get(0)
	vy := y.get(0)
	res.set(0, min(vx, vy))
	vx = x.get(1)
	vy = y.get(1)
	res.set(1, min(vx, vy))
	return res
}

// Mul returns the elementwise product of x and y.
//
//	z[i] = x[i] * y[i]
func (x Float64s) Mul(y Float64s) (z Float64s) {
	var res Float64s
	res.set(0, x.get(0)*y.get(0))
	res.set(1, x.get(1)*y.get(1))
	return res
}

// MulAdd returns x * y + z elementwise.
//
//	w[i] = x[i] * y[i] + z[i]
func (x Float64s) MulAdd(y Float64s, z Float64s) (w Float64s) {
	var res Float64s
	res.set(0, x.get(0)*y.get(0)+z.get(0))
	res.set(1, x.get(1)*y.get(1)+z.get(1))
	return res
}

// Neg returns the elementwise negation of x.
//
//	z[i] = -x[i]
func (x Float64s) Neg() (z Float64s) {
	var res Float64s
	for i := 0; i < 4; i++ {
		res.set(i, -(x.get(i)))
	}
	return res
}

// NotEqual returns a mask indicating which elements of x and y are not equal.
func (x Float64s) NotEqual(y Float64s) (z Mask64s) {
	var res Mask64s
	if x.get(0) != y.get(0) {
		res.a = ^uint64(0)
	}
	if x.get(1) != y.get(1) {
		res.b = ^uint64(0)
	}
	return res
}

// ReduceSum returns the scalar sum of the elements of x.
//
//	z = x[0] + x[1] + ...
func (x Float64s) ReduceSum() (z float64) {
	var res float64
	for i := 0; i < 2; i++ {
		res += x.get(i)
	}
	return res
}

// Sqrt returns the elementwise square root of x.
//
//	z[i] = sqrt(x[i])
func (x Float64s) Sqrt() (z Float64s) {
	var res Float64s
	res.set(0, math.Sqrt(x.get(0)))
	res.set(1, math.Sqrt(x.get(1)))
	return res
}

// Store stores the elements of x into a slice. If len(s) < x.Len(), it panics.
func (x Float64s) Store(s []float64) {
	if len(s) > 0 {
		s[0] = x.get(0)
	}
	if len(s) > 1 {
		s[1] = x.get(1)
	}
}

// StorePart stores n=min(len(s), x.Len()) elements of x into s and returns n.
func (x Float64s) StorePart(s []float64) (n int) {
	x.Store(s)
	return min(len(s), x.Len())
}

// String returns a string representation of the vector.
func (x Float64s) String() string {
	return sliceToString([]float64{x.get(0), x.get(1)})
}

// Sub subtracts y from x elementwise.
//
//	z[i] = x[i] - y[i]
func (x Float64s) Sub(y Float64s) (z Float64s) {
	var res Float64s
	res.set(0, x.get(0)-y.get(0))
	res.set(1, x.get(1)-y.get(1))
	return res
}

// ToBits returns the IEEE 754 binary representation of each element of x.
func (x Float64s) ToBits() (z Uint64s) {
	return Uint64s{a: x.a, b: x.b}
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
//
//	z[i] = x[i] & y[i]
func (x Mask8s) And(y Mask8s) (z Mask8s) {
	return Mask8s{a: x.a & y.a, b: x.b & y.b}
}

// Or returns the bitwise OR of x and y.
//
//	z[i] = x[i] | y[i]
func (x Mask8s) Or(y Mask8s) (z Mask8s) {
	return Mask8s{a: x.a | y.a, b: x.b | y.b}
}

// String returns a string representation of the vector.
func (x Mask8s) String() string {
	var s [16]int8
	x.ToInt8s().Neg().Store(s[:])
	return sliceToString(s[:])
}

// ToInt8s converts the mask to a vector, where element i is set to ^0 (all bits
// set, e.g., -1) if mask element i is "true", and 0 otherwise.
func (x Mask8s) ToInt8s() (z Int8s) {
	return Int8s{a: x.a, b: x.b}
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
//
//	z[i] = x[i] & y[i]
func (x Mask16s) And(y Mask16s) (z Mask16s) {
	return Mask16s{a: x.a & y.a, b: x.b & y.b}
}

// Or returns the bitwise OR of x and y.
//
//	z[i] = x[i] | y[i]
func (x Mask16s) Or(y Mask16s) (z Mask16s) {
	return Mask16s{a: x.a | y.a, b: x.b | y.b}
}

// String returns a string representation of the vector.
func (x Mask16s) String() string {
	var s [8]int16
	x.ToInt16s().Neg().Store(s[:])
	return sliceToString(s[:])
}

// ToInt16s converts the mask to a vector, where element i is set to ^0 (all bits
// set, e.g., -1) if mask element i is "true", and 0 otherwise.
func (x Mask16s) ToInt16s() (z Int16s) {
	return Int16s{a: x.a, b: x.b}
}

func (x *Mask32s) set(i int, v bool) {
	if v {
		if i < 2 {
			mask := uint64(0xffff_ffff) << (32 * i)
			x.a |= mask
		} else {
			mask := uint64(0xffff_ffff) << (32 * (i - 2))
			x.b |= mask
		}
	}
}

// And returns the bitwise AND of x and y.
//
//	z[i] = x[i] & y[i]
func (x Mask32s) And(y Mask32s) (z Mask32s) {
	return Mask32s{a: x.a & y.a, b: x.b & y.b}
}

// Or returns the bitwise OR of x and y.
//
//	z[i] = x[i] | y[i]
func (x Mask32s) Or(y Mask32s) (z Mask32s) {
	return Mask32s{a: x.a | y.a, b: x.b | y.b}
}

// String returns a string representation of the vector.
func (x Mask32s) String() string {
	var s [4]int32
	x.ToInt32s().Neg().Store(s[:])
	return sliceToString(s[:])
}

// ToInt32s converts the mask to a vector, where element i is set to ^0 (all bits
// set, e.g., -1) if mask element i is "true", and 0 otherwise.
func (x Mask32s) ToInt32s() (z Int32s) {
	return Int32s{a: x.a, b: x.b}
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
//
//	z[i] = x[i] & y[i]
func (x Mask64s) And(y Mask64s) (z Mask64s) {
	return Mask64s{a: x.a & y.a, b: x.b & y.b}
}

// Or returns the bitwise OR of x and y.
//
//	z[i] = x[i] | y[i]
func (x Mask64s) Or(y Mask64s) (z Mask64s) {
	return Mask64s{a: x.a | y.a, b: x.b | y.b}
}

// String returns a string representation of the vector.
func (x Mask64s) String() string {
	var s [2]int64
	x.ToInt64s().Neg().Store(s[:])
	return sliceToString(s[:])
}

// ToInt64s converts the mask to a vector, where element i is set to ^0 (all bits
// set, e.g., -1) if mask element i is "true", and 0 otherwise.
func (x Mask64s) ToInt64s() (z Int64s) {
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
	m0 = newT(0x1084210842108421, 0x2108421084210842)
	m1 = newT(0x2108421084210842, 0x4210842108421084)
	m2 = newT(0x4210842108421084, 0x8421084210842108)
	m3 = newT(0x8421084210842108, 0x0842108421084210)
	m4 = newT(0x0842108421084210, 0x1084210842108421)
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

// CarrylessMultiplyEven computes the elementwise carryless multiplication of
// even-indexed elements of x and y. The result of each carryless multiply is
// twice the width of the input elements. The high bits are stored in elements
// z[2*i+1] and the low bits are stored in elements z[2*i].
//
//	concat(z[2*i+1], z[2*i]) = clmul(x[2*i], y[2*i])
//
// A carryless multiplication uses bitwise XOR instead of add-with-carry.
// For example, to compute the carryless multiply of 0b1110 and 0b1011:
//
//	     1110
//	   ⊗ 1011
//	  ───────
//	     1110
//	    1110
//	   0000
//	⊕ 1110
//	  ───────
//	  1100010
//
// Carryless multiply can also be viewed as multiplying polynomials with
// coefficients from GF(2). For example, the above example can be represented as
//
//	  (x^3 + x^2 + x^1) * (x^3 + x^1 + x^0)
//	= (x^6 + x^5 + (1^1)x^4 + (1^1)x^3 + (1^1)x^2 + x^1)
//	= (x^6 + x^5 + x^1)
func (x Uint64s) CarrylessMultiplyEven(y Uint64s) (z Uint64s) {
	return x.clmul(y)
}

// CarrylessMultiplyOdd computes the elementwise carryless multiplication of
// odd-indexed elements of x and y. The result of each carryless multiply is
// twice the width of the input elements. The high bits are stored in elements
// z[2*i+1] and the low bits are stored in elements z[2*i].
//
//	concat(z[2*i+1], z[2*i]) = clmul(x[2*i+1], y[2*i+1])
//
// See [CarrylessMultiplyEven] for details about carryless multiply.
func (x Uint64s) CarrylessMultiplyOdd(y Uint64s) (z Uint64s) {
	x.a = x.b
	y.a = y.b
	return x.clmul(y)
}

// OnesCount counts the number of one bits ("population count") in
// each element.
//
//	z[i] = bits.OnesCount(x[i])
func (x Int8s) OnesCount() (z Int8s) {
	a0, a1 := x.a, x.b
	m1 := uint64(0x5555555555555555)
	m2 := uint64(0x3333333333333333)
	m4 := uint64(0x0f0f0f0f0f0f0f0f)
	a0 = (a0 & m1) + ((a0 >> 1) & m1)
	a1 = (a1 & m1) + ((a1 >> 1) & m1)

	a0 = (a0 & m2) + ((a0 >> 2) & m2)
	a1 = (a1 & m2) + ((a1 >> 2) & m2)

	a0 = (a0 & m4) + ((a0 >> 4) & m4)
	a1 = (a1 & m4) + ((a1 >> 4) & m4)

	return Int8s{a: a0, b: a1}
}

// OnesCount counts the number of one bits ("population count") in
// each element.
//
//	z[i] = bits.OnesCount(x[i])
func (x Uint8s) OnesCount() (z Uint8s) {
	a0, a1 := x.a, x.b
	m1 := uint64(0x5555555555555555)
	m2 := uint64(0x3333333333333333)
	m4 := uint64(0x0f0f0f0f0f0f0f0f)
	a0 = (a0 & m1) + ((a0 >> 1) & m1)
	a1 = (a1 & m1) + ((a1 >> 1) & m1)

	a0 = (a0 & m2) + ((a0 >> 2) & m2)
	a1 = (a1 & m2) + ((a1 >> 2) & m2)

	a0 = (a0 & m4) + ((a0 >> 4) & m4)
	a1 = (a1 & m4) + ((a1 >> 4) & m4)

	return Uint8s{a: a0, b: a1}
}

const (
	by8  = 0x0101010101010101
	by16 = 0x0001000100010001
)

// BroadcastInt8s returns a vector with the input x assigned to all elements of the
// result.
func BroadcastInt8s(x int8) (z Int8s) {
	v := (255 & uint64(x)) * by8
	return Int8s{a: v, b: v}
}

// BroadcastInt16s returns a vector with the input x assigned to all elements of the
// result.
func BroadcastInt16s(x int16) (z Int16s) {
	v := (65535 & uint64(x)) * by16
	return Int16s{a: v, b: v}
}

// BroadcastInt32s returns a vector with the input x assigned to all elements of the
// result.
func BroadcastInt32s(x int32) (z Int32s) {
	v := uint64(x) & 0xffff_ffff
	v = v<<32 | v
	return Int32s{a: v, b: v}
}

// BroadcastInt64s returns a vector with the input x assigned to all elements of the
// result.
func BroadcastInt64s(x int64) (z Int64s) {
	v := uint64(x)
	return Int64s{a: v, b: v}
}

// BroadcastUint8s returns a vector with the input x assigned to all elements of the
// result.
func BroadcastUint8s(x uint8) (z Uint8s) {
	v := uint64(x) * by8
	return Uint8s{a: v, b: v}

}

// BroadcastUint16s returns a vector with the input x assigned to all elements of the
// result.
func BroadcastUint16s(x uint16) (z Uint16s) {
	v := uint64(x) * by16
	return Uint16s{a: v, b: v}

}

// BroadcastUint32s returns a vector with the input x assigned to all elements of the
// result.
func BroadcastUint32s(x uint32) (z Uint32s) {
	v := uint64(x)
	v = v<<32 | v
	return Uint32s{a: v, b: v}
}

// BroadcastUint64s returns a vector with the input x assigned to all elements of the
// result.
func BroadcastUint64s(x uint64) (z Uint64s) {
	return Uint64s{a: x, b: x}
}

// BroadcastFloat32s returns a vector with the input x assigned to all elements of the
// result.
func BroadcastFloat32s(x float32) (z Float32s) {
	v := uint64(math.Float32bits(x))
	v = v<<32 | v
	return Float32s{a: v, b: v}
}

// BroadcastFloat64s returns a vector with the input x assigned to all elements of the
// result.
func BroadcastFloat64s(x float64) (z Float64s) {
	v := math.Float64bits(x)
	return Float64s{a: v, b: v}
}

// All returns true when all positions in mask x are true.
func (x Mask8s) All() bool {
	return x.a&x.b == 0xffff_ffff_ffff_ffff
}

// Any returns true when any position in mask x is true.
func (x Mask8s) Any() bool {
	return x.a|x.b != 0
}

// None returns true when no positions in mask x are set.
func (x Mask8s) None() bool {
	return x.a|x.b == 0
}

// All returns true when all positions in mask x are true.
func (x Mask16s) All() bool {
	return x.a&x.b == 0xffff_ffff_ffff_ffff
}

// Any returns true when any position in mask x is true.
func (x Mask16s) Any() bool {
	return x.a|x.b != 0
}

// None returns true when no positions in mask x are set.
func (x Mask16s) None() bool {
	return x.a|x.b == 0
}

// All returns true when all positions in mask x are true.
func (x Mask32s) All() bool {
	return x.a&x.b == 0xffff_ffff_ffff_ffff
}

// Any returns true when any position in mask x is true.
func (x Mask32s) Any() bool {
	return x.a|x.b != 0
}

// None returns true when no positions in mask x are set.
func (x Mask32s) None() bool {
	return x.a|x.b == 0
}

// All returns true when all positions in mask x are true.
func (x Mask64s) All() bool {
	return x.a&x.b == 0xffff_ffff_ffff_ffff
}

// Any returns true when any position in mask x is true.
func (x Mask64s) Any() bool {
	return x.a|x.b != 0
}

// None returns true when no positions in mask x are set.
func (x Mask64s) None() bool {
	return x.a|x.b == 0
}

// TrailingZeros returns the number of low-order false (zero) elements in mask
// x. If the mask is entirely false, it returns x.Len().
func (x Mask8s) TrailingZeros() int {
	a0 := x.a
	a := a0 & 0x0101010101010101
	lane := bits.TrailingZeros64(a)
	if lane < 64 {
		return lane >> 3
	}
	a0 = x.b
	a = a0 & 0x0101010101010101
	lane = bits.TrailingZeros64(a)
	return lane>>3 + 8
}

// TrailingZeros returns the number of low-order false (zero) elements in mask
// x. If the mask is entirely false, it returns x.Len().
func (x Mask16s) TrailingZeros() int {
	a0 := x.a
	a := a0 & 0x0001000100010001
	lane := bits.TrailingZeros64(a)
	if lane < 64 {
		return lane >> 4
	}
	a0 = x.b
	a = a0 & 0x0001000100010001
	lane = bits.TrailingZeros64(a)
	return lane>>4 + 4
}

// TrailingZeros returns the number of low-order false (zero) elements in mask
// x. If the mask is entirely false, it returns x.Len().
func (x Mask32s) TrailingZeros() int {
	a0 := x.a
	a := a0 & 0x0000000100000001
	lane := bits.TrailingZeros64(a)
	if lane < 64 {
		return lane >> 5
	}
	a0 = x.b
	a = a0 & 0x0000000100000001
	lane = bits.TrailingZeros64(a)
	return lane>>5 + 2
}

// TrailingZeros returns the number of low-order false (zero) elements in mask
// x. If the mask is entirely false, it returns x.Len().
func (x Mask64s) TrailingZeros() int {
	if x.a != 0 {
		return 0
	}
	if x.b != 0 {
		return 1
	}
	return 2
}
