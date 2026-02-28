// asmcheck

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that we are zeroing directly instead of
// copying a large zero value. Issue 38554.

package codegen

func retlarge() [256]byte {
	// amd64:-"DUFFCOPY"
	return [256]byte{}
}
