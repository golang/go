// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// // No C code required.
import "C"

import "common"

func ReadCommonX() int {
	return common.X
}

var Seven int

func init() {
	Seven = 7
	common.X *= Seven
}

func main() {
	panic("plugin1.main called")
}
