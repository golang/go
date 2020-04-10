// asmcheck

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

func efaceExtract(e interface{}) int {
	// This should be compiled with only
	// a single conditional jump.
	// amd64:-"JMP"
	if x, ok := e.(int); ok {
		return x
	}
	return 0
}
