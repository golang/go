// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Windows environment variables.

package syscall

import (
	"unicode/utf16"
	"unsafe"
)

func Getenv(key string) (value string, found bool) {
	b := make([]uint16, 100)
	n, e := GetEnvironmentVariable(StringToUTF16Ptr(key), &b[0], uint32(len(b)))
	if n == 0 && e == ERROR_ENVVAR_NOT_FOUND {
		return "", false
	}
	if n > uint32(len(b)) {
		b = make([]uint16, n)
		n, e = GetEnvironmentVariable(StringToUTF16Ptr(key), &b[0], uint32(len(b)))
		if n > uint32(len(b)) {
			n = 0
		}
	}
	if n == 0 {
		return "", false
	}
	return string(utf16.Decode(b[0:n])), true
}

func Setenv(key, value string) error {
	var v *uint16
	if len(value) > 0 {
		v = StringToUTF16Ptr(value)
	}
	e := SetEnvironmentVariable(StringToUTF16Ptr(key), v)
	if e != nil {
		return e
	}
	return nil
}

func Clearenv() {
	for _, s := range Environ() {
		// Environment variables can begin with =
		// so start looking for the separator = at j=1.
		// http://blogs.msdn.com/b/oldnewthing/archive/2010/05/06/10008132.aspx
		for j := 1; j < len(s); j++ {
			if s[j] == '=' {
				Setenv(s[0:j], "")
				break
			}
		}
	}
}

func Environ() []string {
	s, e := GetEnvironmentStrings()
	if e != nil {
		return nil
	}
	defer FreeEnvironmentStrings(s)
	r := make([]string, 0, 50) // Empty with room to grow.
	for from, i, p := 0, 0, (*[1 << 24]uint16)(unsafe.Pointer(s)); true; i++ {
		if p[i] == 0 {
			// empty string marks the end
			if i <= from {
				break
			}
			r = append(r, string(utf16.Decode(p[from:i])))
			from = i + 1
		}
	}
	return r
}
