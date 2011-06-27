// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bufio"
	"./io"
	goio "io"
)

func main() {
	// The errors here complain that io.X != io.X
	// for different values of io so they should be
	// showing the full import path, which for the
	// "./io" import is really ..../go/test/io.
	// For example:
	//
	// main.go:25: cannot use w (type "/Users/rsc/g/go/test/fixedbugs/bug345.dir/io".Writer) as type "io".Writer in function argument:
	//	io.Writer does not implement io.Writer (missing Write method)
	// main.go:27: cannot use &x (type *"io".SectionReader) as type *"/Users/rsc/g/go/test/fixedbugs/bug345.dir/io".SectionReader in function argument

	var w io.Writer
	bufio.NewWriter(w)  // ERROR "test/io"
	var x goio.SectionReader
	io.SR(&x)  // ERROR "test/io"
}
