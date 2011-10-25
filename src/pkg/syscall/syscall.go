// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package syscall contains an interface to the low-level operating system
// primitives.  The details vary depending on the underlying system.
// Its primary use is inside other packages that provide a more portable
// interface to the system, such as "os", "time" and "net".  Use those
// packages rather than this one if you can.
// For details of the functions and data types in this package consult
// the manuals for the appropriate operating system.
// These calls return errno == 0 to indicate success; otherwise
// errno is an operating system error number describing the failure.
package syscall

// StringByteSlice returns a NUL-terminated slice of bytes
// containing the text of s.
func StringByteSlice(s string) []byte {
	a := make([]byte, len(s)+1)
	copy(a, s)
	return a
}

// StringBytePtr returns a pointer to a NUL-terminated array of bytes
// containing the text of s.
func StringBytePtr(s string) *byte { return &StringByteSlice(s)[0] }

// Single-word zero for use when we need a valid pointer to 0 bytes.
// See mksyscall.pl.
var _zero uintptr
