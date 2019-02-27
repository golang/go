// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// // No C code required.
import "C"

// The common package imported here does not match the common package
// imported by plugin1. A program that attempts to load plugin1 and
// plugin-mismatch should produce an error.
import "testplugin/common"

func ReadCommonX() int {
	return common.X
}
