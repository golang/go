// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file is processed by the cover command, then a test verifies that
// all compiler directives are preserved and positioned appropriately.

//go:a

//go:b
package main

//go:c1

//go:c2
//doc
func c() {
}

//go:d1

//doc
//go:d2
type d int

//go:e1

//doc
//go:e2
type (
	e int
	f int
)

//go:_empty1
//doc
//go:_empty2
type ()

//go:f
