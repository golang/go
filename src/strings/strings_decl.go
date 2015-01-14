// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strings

// IndexByte returns the index of the first instance of c in s, or -1 if c is not present in s.
func IndexByte(s string, c byte) int // ../runtime/asm_$GOARCH.s

// Compare returns an integer comparing two strings lexicographically.
// The result will be 0 if a==b, -1 if a < b, and +1 if a > b.
//
// In most cases it is simpler to use the built-in comparison operators
// ==, <, >, and so on.
func Compare(a, b string) int // ../runtime/noasm.go or ../runtime/asm_*.s
