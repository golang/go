// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bytes

//go:noescape

// IndexByte returns the index of the first instance of c in s, or -1 if c is not present in s.
func IndexByte(s []byte, c byte) int // asm_$GOARCH.s

//go:noescape

// Equal returns a boolean reporting whether a == b.
// A nil argument is equivalent to an empty slice.
func Equal(a, b []byte) bool // asm_arm.s or ../runtime/asm_{386,amd64}.s
