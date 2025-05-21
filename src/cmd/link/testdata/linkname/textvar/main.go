// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Using a linknamed variable to reference an assembly
// function in the same package is ok.

package main

import _ "unsafe"

func main() {
	println(&asmfunc)
}

//go:linkname asmfunc
var asmfunc uintptr
