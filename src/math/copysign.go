// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math

// Copysign returns a value with the magnitude of f
// and the sign of sign.
func Copysign(f, sign float64) float64 {
	const signBit = 1 << 63
	return Float64frombits(Float64bits(f)&^signBit | Float64bits(sign)&signBit)
}
