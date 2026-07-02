// asmcheck

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

//go:noinline
func use16(b []byte, x uint64) {
	b[0] = byte(x)
}

//go:noinline
func resultSlotArray(x uint64) [16]byte {
	var ret [16]byte
	// amd64:-`.*\.ret\+`
	use16(ret[:8], x)
	// amd64:-`.*\.ret\+`
	use16(ret[8:], x)
	// amd64:-`.*\.ret\+` -`.*MOVUPS.*~r0`
	return ret
}

//go:noinline
func resultSlotArrayMultipleReturns(x uint64, c bool) [16]byte {
	var ret [16]byte
	// amd64:-`.*\.ret\+`
	use16(ret[:8], x)
	if c {
		// amd64:-`.*\.ret\+` -`.*MOVUPS.*~r0`
		return ret
	}
	// amd64:-`.*\.ret\+`
	use16(ret[8:], x)
	// amd64:-`.*\.ret\+` -`.*MOVUPS.*~r0`
	return ret
}
