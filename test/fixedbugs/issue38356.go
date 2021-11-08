// compile

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Make sure floating point operations that generate flags
// are scheduled correctly on s390x.

package p

func f1(x, y float64, z int) float64 {
	a := x + y  // generate flags
	if z == 0 { // create basic block that does not clobber flags
		return a
	}
	if a > 0 { // use flags in different basic block
		return y
	}
	return x
}

func f2(x, y float64, z int) float64 {
	a := x - y  // generate flags
	if z == 0 { // create basic block that does not clobber flags
		return a
	}
	if a > 0 { // use flags in different basic block
		return y
	}
	return x
}

func f3(x, y float32, z int) float32 {
	a := x + y  // generate flags
	if z == 0 { // create basic block that does not clobber flags
		return a
	}
	if a > 0 { // use flags in different basic block
		return y
	}
	return x
}

func f4(x, y float32, z int) float32 {
	a := x - y  // generate flags
	if z == 0 { // create basic block that does not clobber flags
		return a
	}
	if a > 0 { // use flags in different basic block
		return y
	}
	return x
}
