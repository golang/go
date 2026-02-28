// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build cgo

// Issue 8828: compiling a file with -compiler=gccgo fails if a .c file
// has the same name as compiled directory.

package cgotest

import "cmd/cgo/internal/test/issue8828"

func p() {
	issue8828.Bar()
}
