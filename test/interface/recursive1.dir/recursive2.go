// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that the mutually recursive types in recursive1.go made it
// intact and with the same meaning, by assigning to or using them.

package main

import "./recursive1"

func main() {
	var i1 p.I1
	var i2 p.I2
	i1 = i2
	i2 = i1
	i1 = i2.F()
	i2 = i1.F()
	_, _ = i1, i2
}
