// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build wasm || wasm32

package main

import (
	"reflect"
	"runtime"
	"unsafe"
)

// A runtime method has a low logical PC. Its code pointer must remain valid
// when stack growth adjusts pointers in the caller's frame.
func mapError() (err any) {
	defer func() { err = recover() }()
	var m map[any]int
	_ = m[[]int{}]
	return
}

//go:noinline
func grow(n int, p unsafe.Pointer) {
	var pad [512]byte
	pad[n%len(pad)] = byte(n)
	if n > 0 {
		grow(n-1, p)
	}
	runtime.KeepAlive(p)
	runtime.KeepAlive(&pad)
}

func main() {
	m, ok := reflect.TypeOf(mapError()).MethodByName("Error")
	if !ok {
		panic("missing Error method")
	}
	p := m.Func.UnsafePointer()
	if runtime.FuncForPC(uintptr(p)) == nil {
		panic("missing function for method PC")
	}
	grow(1000, p)
	runtime.KeepAlive(p)
}
