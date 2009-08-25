// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gmp

import "os"

type Int struct {
	hidden *byte
}

func addInt(z, x, y *Int) *Int
func stringInt(z *Int) string
func divInt(z, x, y *Int) *Int
func mulInt(z, x, y *Int) *Int
func subInt(z, x, y *Int) *Int
func modInt(z, x, y *Int) *Int
func rshInt(z, x *Int, s uint) *Int
func lshInt(z, x *Int, s uint) *Int
func expInt(z, x, y, m *Int) *Int
func lenInt(z *Int) int
func bytesInt(z *Int) []byte
func setInt(z *Int, x *Int) *Int
func setBytesInt(z *Int, b []byte) *Int
func setStringInt(z *Int, s string, b int) int
func setInt64Int(z *Int, x int64) *Int
func int64Int(z *Int) int64

// NewInt returns a new Int initialized to x.
func NewInt(x int64) *Int

// z = x + y
func (z *Int) Add(x, y *Int) *Int {
	return addInt(z, x, y)
}

// z = x - y
func (z *Int) Sub(x, y *Int) *Int {
	return subInt(z, x, y)
}

// z = x * y
func (z *Int) Mul(x, y *Int) *Int {
	return mulInt(z, x, y)
}

// z = x
func (z *Int) SetInt64(x int64) *Int {
	return setInt64Int(z, x);
}

// z = x / y
func (z *Int) Div(x, y *Int) *Int {
	return divInt(z, x, y)
}

// z = x % y
func (z *Int) Mod(x, y *Int) *Int {
	return modInt(z, x, y)
}

// z = x^y if m == nil, x^y % m otherwise
func (z *Int) Exp(x, y, m *Int) *Int {
	return expInt(z, x, y, m);
}

// z = x << s
func (z *Int) Lsh(x *Int, s uint) *Int {
	return lshInt(z, x, s);
}

// z = x >> s
func (z *Int) Rsh(x *Int, s uint) *Int {
	return rshInt(z, x, s);
}

// z = x
func (z *Int) Set(x *Int) *Int {
	return setInt(z, x);
}

// Len returns length of z in bits.
func (z *Int) Len() int {
	return lenInt(z);
}

func (z *Int) String() string {
	return stringInt(z)
}

func (z *Int) Int64() int64 {
	return int64Int(z)
}

// TODO: better name?  Maybe return []byte instead?
// Bytes writes a big-endian representation of z into b.
// If b is not large enough to contain all of z, the lowest
// bits are stored.
func (z *Int) Bytes() []byte {
	return bytesInt(z);
}

// SetBytes sets z to the integer represented by the bytes of b
// interpreted as a big-endian integer.
func (z *Int) SetBytes(b []byte) *Int {
	return setBytesInt(z, b);
}

// SetString parses the string s in base b (8, 10, 16) and sets z to the result.
// It returns an error if the string cannot be parsed or the base is invalid.
func (z *Int) SetString(s string, b int) os.Error {
	if b <= 0 || b > 36 || setStringInt(z, s, b) < 0 {
		return os.EINVAL;
	}
	return nil;
}

// GcdInt sets d to the greatest common divisor of a and b
// and sets x and y such that d = a*x + b*y.
// The inputs a and b must be positive.
// Pass x == nil and y == nil if only d is needed.
// If a <= 0 or b <= 0, GcdInt sets d, x, and y to zero.
func GcdInt(d, x, y, a, b *Int)

// CmpInt compares x and y.  The result is -1, 0, +1.
func CmpInt(x, y *Int) int

// DivModInt sets q = x/y, r = x%y.
func DivModInt(q, r, x, y *Int)
