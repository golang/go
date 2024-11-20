// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

func gogetenv(key string) string {
	env := environ()
	if env == nil {
		throw("getenv before env init")
	}
	for _, s := range env {
		if len(s) > len(key) && s[len(key)] == '=' && envKeyEqual(s[:len(key)], key) {
			return s[len(key)+1:]
		}
	}
	return ""
}

// envKeyEqual reports whether a == b, with ASCII-only case insensitivity
// on Windows. The two strings must have the same length.
func envKeyEqual(a, b string) bool {
	if GOOS == "windows" { // case insensitive
		for i := 0; i < len(a); i++ {
			ca, cb := a[i], b[i]
			if ca == cb || lowerASCII(ca) == lowerASCII(cb) {
				continue
			}
			return false
		}
		return true
	}
	return a == b
}

func lowerASCII(c byte) byte {
	if 'A' <= c && c <= 'Z' {
		return c + ('a' - 'A')
	}
	return c
}

// _cgo_setenv should be an internal detail,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - github.com/ebitengine/purego
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname _cgo_setenv
var _cgo_setenv unsafe.Pointer // pointer to C function

// _cgo_unsetenv should be an internal detail,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - github.com/ebitengine/purego
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname _cgo_unsetenv
var _cgo_unsetenv unsafe.Pointer // pointer to C function

// Update the C environment if cgo is loaded.
func setenv_c(k string, v string) {
	if _cgo_setenv == nil {
		return
	}
	arg := [2]unsafe.Pointer{cstring(k), cstring(v)}
	asmcgocall(_cgo_setenv, unsafe.Pointer(&arg))
}

// Update the C environment if cgo is loaded.
func unsetenv_c(k string) {
	if _cgo_unsetenv == nil {
		return
	}
	arg := [1]unsafe.Pointer{cstring(k)}
	asmcgocall(_cgo_unsetenv, unsafe.Pointer(&arg))
}

func cstring(s string) unsafe.Pointer {
	p := make([]byte, len(s)+1)
	copy(p, s)
	return unsafe.Pointer(&p[0])
}
