// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build js

package runtime

import "unsafe"

// resetMemoryDataView signals the JS front-end that WebAssembly's memory.grow instruction has been used.
// This allows the front-end to replace the old DataView object with a new one.
//
//go:wasmimport gojs runtime.resetMemoryDataView
func resetMemoryDataView()

// allocErrorString allocates a space to store the error string passed from the JS front-end.
//
//go:wasmexport allocErrorString
func allocErrorString(s int64) uintptr {
	tmp := make([]byte, s)
	return uintptr(unsafe.Pointer(&tmp[0]))
}
