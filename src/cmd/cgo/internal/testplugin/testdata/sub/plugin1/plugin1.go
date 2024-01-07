// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// // No C code required.
import "C"

import "testplugin/common"

func F() int { return 17 }

var FuncVar = func() {}

func ReadCommonX() int {
	FuncVar()
	return common.X
}

func main() {
	panic("plugin1.main called")
}
