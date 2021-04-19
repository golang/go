// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Windows environment variables.

package syscall

import (
	"unicode/utf16"
	"unsafe"
)

func Getenv(key string) (value string, found bool) {
	keyp, err := UTF16PtrFromString(key)
	if err != nil {
		return "", false
	}
	n := uint32(100)
	for {
		b := make([]uint16, n)
		n, err = GetEnvironmentVariable(keyp, &b[0], uint32(len(b)))
		if n == 0 && err == ERROR_ENVVAR_NOT_FOUND {
			return "", false
		}
		if n <= uint32(len(b)) {
			return string(utf16.Decode(b[:n])), true
		}
	}
}

func Setenv(key, value string) error {
	v, err := UTF16PtrFromString(value)
	if err != nil {
		return err
	}
	keyp, err := UTF16PtrFromString(key)
	if err != nil {
		return err
	}
	e := SetEnvironmentVariable(keyp, v)
	if e != nil {
		return e
	}
	return nil
}

func Unsetenv(key string) error {
	keyp, err := UTF16PtrFromString(key)
	if err != nil {
		return err
	}
	return SetEnvironmentVariable(keyp, nil)
}

func Clearenv() {
	for _, s := range Environ() {
		// Environment variables can begin with =
		// so start looking for the separator = at j=1.
		// http://blogs.msdn.com/b/oldnewthing/archive/2010/05/06/10008132.aspx
		for j := 1; j < len(s); j++ {
			if s[j] == '=' {
				Unsetenv(s[0:j])
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
