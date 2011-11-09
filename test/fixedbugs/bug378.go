// errchk $G $D/$F.go

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 1387
package foo

import "bytes"

func i() {
	a := make([]bytes.Buffer, 1)
	b := a[0] // ERROR "unexported field"
}

func f() {
	a := make([]bytes.Buffer, 1)
	a = append(a, a...) // ERROR "unexported field"
}


func g() {
	a := make([]bytes.Buffer, 1)
	b := make([]bytes.Buffer, 1)
	copy(b, a)	// ERROR "unexported field"
}
