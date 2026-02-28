// errorcheck

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// selector expression resolves incorrectly for defined
// pointer types.

package main

type E struct{ f int }
type T struct{ E }

func (*T) f() int { return 0 }

type P *T
type PP **T

func main() {
	var x P
	_ = x.f // ERROR "x\.f undefined \(type P has no field or method f\)"

	var y PP
	_ = y.f // ERROR "y\.f undefined \(type PP has no field or method f\)"
}
