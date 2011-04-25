// $G $D/$F.go || echo BUG: bug334

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 1716

package main

type (
	cplx64  complex64
	cplx128 complex128
)

func (c cplx64) Foo()  {}
func (c cplx128) Foo() {}

func main() {
	var c64 cplx128
	var c128 cplx64
	c64.Foo()
	c128.Foo()
}

/*
bug334.go:16: invalid receiver type cplx64
bug334.go:17: invalid receiver type cplx128
bug334.go:22: c64.Foo undefined (type cplx128 has no field or method Foo)
bug334.go:23: c128.Foo undefined (type cplx64 has no field or method Foo)
*/
