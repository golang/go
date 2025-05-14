// asmcheck

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

func f(n int) int {
	r := 0
	// arm64:-"MOVD\t R"
	// amd64:-"LEAQ","INCQ"
	for i := range n {
		r += i
	}
	return r
}
