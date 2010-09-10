// errchk $G $D/$F.go

// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	_ = asdf	// ERROR "undefined.*asdf"

	new = 1	// ERROR "use of builtin new not in function call|invalid left hand side"
}

