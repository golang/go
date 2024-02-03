// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Windows environment variables.

package syscall

import (
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
			return UTF16ToString(b[:n]), true
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
	runtimeSetenv(key, value)
	return nil
}

func Unsetenv(key string) error {
	keyp, err := UTF16PtrFromString(key)
	if err != nil {
		return err
	}
	e := SetEnvironmentVariable(keyp, nil)
	if e != nil {
		return e
	}
	runtimeUnsetenv(key)
	return nil
}

func Clearenv() {
	for _, s := range Environ() {
		// Environment variables can begin with =
		// so start looking for the separator = at j=1.
		// https://devblogs.microsoft.com/oldnewthing/20100506-00/?p=14133
		for j := 1; j < len(s); j++ {
			if s[j] == '=' {
				Unsetenv(s[0:j])
				break
			}
		}
	}
}

func Environ() []string {
	envp, e := GetEnvironmentStrings()
	if e != nil {
		return nil
	}
	defer FreeEnvironmentStrings(envp)

	r := make([]string, 0, 50) // Empty with room to grow.
	const size = unsafe.Sizeof(*envp)
	for *envp != 0 { // environment block ends with empty string
		// find NUL terminator
		end := unsafe.Pointer(envp)
		for *(*uint16)(end) != 0 {
			end = unsafe.Add(end, size)
		}

		entry := unsafe.Slice(envp, (uintptr(end)-uintptr(unsafe.Pointer(envp)))/size)
		r = append(r, UTF16ToString(entry))
		envp = (*uint16)(unsafe.Add(end, size))
	}
	return r
}
