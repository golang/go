// run

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Do not panic on conversion to anonymous interface, which
// is similar-looking interface types in different packages.

package main

import (
	ssa1 "./p1/ssa"
	ssa2 "./p2/ssa"
)

func main() {
	v1 := &ssa1.T{}
	_ = v1

	v2 := &ssa2.T{}
	ssa2.Works(v2)
	ssa2.Panics(v2) // This call must not panic
}
