// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sys

func	modf(a double) (double, double);
func	frexp(a double) (int, double);
func	ldexp(double, int) double;

func	Inf(n int) double;
func	NaN() double;
func	isInf(arg double, n int) bool;

export	modf, frexp, ldexp
export	NaN, isInf, Inf
