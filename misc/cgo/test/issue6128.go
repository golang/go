// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

// Test handling of #defined names in clang.
// golang.org/issue/6128.

/*
// NOTE: Must use hex, or else a shortcut for decimals
// in cgo avoids trying to pass this to clang.
#define X 0x1
*/
import "C"

func test6128() {
	// nothing to run, just make sure this compiles.
	_ = C.X
}
