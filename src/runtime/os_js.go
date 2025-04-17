// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build js && wasm

package runtime

import (
	"unsafe"
)

func exit(code int32)

func write1(fd uintptr, p unsafe.Pointer, n int32) int32 {
	if fd > 2 {
		throw("runtime.write to fd > 2 is unsupported")
	}
	wasmWrite(fd, p, n)
	return n
}

//go:wasmimport gojs runtime.wasmWrite
//go:noescape
func wasmWrite(fd uintptr, p unsafe.Pointer, n int32)

func usleep(usec uint32) {
	// TODO(neelance): implement usleep
}

//go:wasmimport gojs runtime.getRandomData
//go:noescape
func getRandomData(r []byte)

func readRandom(r []byte) int {
	getRandomData(r)
	return len(r)
}

func goenvs() {
	goenvs_unix()
}
