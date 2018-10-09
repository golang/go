// asmcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

// This file contains code generation tests related to the handling of
// string types.

func CountRunes(s string) int { // Issue #24923
	// amd64:`.*countrunes`
	return len([]rune(s))
}

func ToByteSlice() []byte { // Issue #24698
	// amd64:`LEAQ\ttype\.\[3\]uint8`
	// amd64:`CALL\truntime\.newobject`
	// amd64:-`.*runtime.stringtoslicebyte`
	return []byte("foo")
}
