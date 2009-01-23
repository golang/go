// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math

// implemented in C, in ../../runtime
// perhaps one day the implementations will move here.

func Float32bits(f float32) (b uint32) 
func Float32frombits(b uint32) (f float32) 
func Float64bits(f float64) (b uint64) 
func Float64frombits(b uint64) (f float64) 
func Frexp(f float64) (frac float64, exp int) 
func Inf(sign int32) (f float64) 
func IsInf(f float64, sign int) (is bool) 
func IsNaN(f float64) (is bool) 
func Ldexp(frac float64, exp int) (f float64) 
func Modf(f float64) (integer float64, frac float64) 
func NaN() (f float64) 
