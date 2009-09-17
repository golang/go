// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This package contains an interface to the low-level operating system
// primitives.  The details vary depending on the underlying system.
// Its primary use is inside other packages that provide a more portable
// interface to the system, such as "os", "time" and "net".  Use those
// packages rather than this one if you can.
// For details of the functions and data types in this package consult
// the manuals for the appropriate operating system.
package syscall

func Syscall(trap, a1, a2, a3 uintptr) (r1, r2, err uintptr)
func Syscall6(trap, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2, err uintptr)
func RawSyscall(trap, a1, a2, a3 uintptr) (r1, r2, err uintptr)

// StringByteSlice returns a NUL-terminated slice of bytes
// containing the text of s.
func StringByteSlice(s string) []byte {
	a := make([]byte, len(s)+1);
	for i := 0; i < len(s); i++ {
		a[i] = s[i];
	}
	return a;
}

// StringBytePtr returns a pointer to a NUL-terminated array of bytes
// containing the text of s.
func StringBytePtr(s string) *byte {
	return &StringByteSlice(s)[0];
}
