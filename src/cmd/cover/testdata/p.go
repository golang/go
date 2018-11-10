// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// A package such that there are 3 functions with zero total and covered lines.
// And one with 1 total and covered lines. Reproduces issue #20515.
package p

//go:noinline
func A() {

}

//go:noinline
func B() {

}

//go:noinline
func C() {

}

//go:noinline
func D() int64 {
	return 42
}
