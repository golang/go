// asmcheck

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

func fa(a [2]int) (r [2]int) {
	// amd64:1`MOVUPS[^,]+, X[0-9]+$`,1`MOVUPS\sX[0-9]+,[^\n]+$`
	return a
}

func fb(a [4]int) (r [4]int) {
	// amd64:2`MOVUPS[^,]+, X[0-9]+$`,2`MOVUPS\sX[0-9]+,[^\n]+$`
	return a
}
