// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux nacl netbsd openbsd solaris windows

package runtime

import "unsafe"

func environ() []string

func getenv(s *byte) *byte {
	val := gogetenv(gostringnocopy(s))
	if val == "" {
		return nil
	}
	// Strings found in environment are NUL-terminated.
	return &bytes(val)[0]
}

func gogetenv(key string) string {
	env := environ()
	if env == nil {
		gothrow("getenv before env init")
	}
	for _, s := range environ() {
		if len(s) > len(key) && s[len(key)] == '=' && s[:len(key)] == key {
			return s[len(key)+1:]
		}
	}
	return ""
}

var _cgo_setenv uintptr   // pointer to C function
var _cgo_unsetenv uintptr // pointer to C function

// Update the C environment if cgo is loaded.
// Called from syscall.Setenv.
func syscall_setenv_c(k string, v string) {
	if _cgo_setenv == 0 {
		return
	}
	arg := [2]unsafe.Pointer{cstring(k), cstring(v)}
	asmcgocall(unsafe.Pointer(_cgo_setenv), unsafe.Pointer(&arg))
}

// Update the C environment if cgo is loaded.
// Called from syscall.unsetenv.
func syscall_unsetenv_c(k string) {
	if _cgo_unsetenv == 0 {
		return
	}
	arg := [1]unsafe.Pointer{cstring(k)}
	asmcgocall(unsafe.Pointer(_cgo_unsetenv), unsafe.Pointer(&arg))
}

func cstring(s string) unsafe.Pointer {
	p := make([]byte, len(s)+1)
	sp := (*_string)(unsafe.Pointer(&s))
	memmove(unsafe.Pointer(&p[0]), unsafe.Pointer(sp.str), uintptr(len(s)))
	return unsafe.Pointer(&p[0])
}
