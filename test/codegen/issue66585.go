// asmcheck

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

var x = func() int {
	n := 0
	f(&n)
	return n
}()

func f(p *int) {
	*p = 1
}

var y = 1

// z can be static initialized.
//
// amd64:-"MOVQ"
var z = y
