// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sys

func	modf(a float64) (x float64, y float64);
func	frexp(a float64) (e int, m float64);
func	ldexp(f float64, e int) float64;

func	Inf(n int) float64;
func	NaN() float64;
func	isInf(arg float64, n int) bool;

export	modf, frexp, ldexp
export	NaN, isInf, Inf
