// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sys

func	modf(a double) (x double, y double);
func	frexp(a double) (e int, m double);
func	ldexp(f double, e int) double;

func	Inf(n int) double;
func	NaN() double;
func	isInf(arg double, n int) bool;

export	modf, frexp, ldexp
export	NaN, isInf, Inf
