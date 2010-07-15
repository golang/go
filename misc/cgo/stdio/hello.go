// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"os"
	"stdio"
)

func main() {
	stdio.Stdout.WriteString(stdio.Greeting + "\n")

	l := stdio.Atol("123")
	if l != 123 {
		println("Atol 123: ", l)
		panic("bad atol")
	}

	n, err := stdio.Strtol("asdf", 123)
	if n != 0 || err != os.EINVAL {
		println("Strtol: ", n, err)
		panic("bad atoi2")
	}

	stdio.TestAlign()
	stdio.TestEnum()
}
