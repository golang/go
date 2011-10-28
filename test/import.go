// $G $D/$F.go

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// check that when import gives multiple names
// to a type, they're still all the same type

package main

import _os_ "os"
import "os"
import . "os"

func f(e *os.File)

func main() {
	var _e_ *_os_.File
	var dot *File

	f(_e_)
	f(dot)
}
