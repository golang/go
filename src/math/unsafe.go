// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math

import "unsafe"

// Float32bits returns the IEEE 754 binary representation of f,
// with the sign bit of f and the result in the same bit position.
// Float32bits(Float32frombits(x)) == x.
func Float32bits(f float32) uint32 { return *(*uint32)(unsafe.Pointer(&f)) }

// Float32frombits returns the floating-point number corresponding
// to the IEEE 754 binary representation b, with the sign bit of b
// and the result in the same bit position.
// Float32frombits(Float32bits(x)) == x.
func Float32frombits(b uint32) float32 { return *(*float32)(unsafe.Pointer(&b)) }

// Float64bits returns the IEEE 754 binary representation of f,
// with the sign bit of f and the result in the same bit position,
// and Float64bits(Float64frombits(x)) == x.
func Float64bits(f float64) uint64 { return *(*uint64)(unsafe.Pointer(&f)) }

// Float64frombits returns the floating-point number corresponding
// to the IEEE 754 binary representation b, with the sign bit of b
// and the result in the same bit position.
// Float64frombits(Float64bits(x)) == x.
func Float64frombits(b uint64) float64 { return *(*float64)(unsafe.Pointer(&b)) }
