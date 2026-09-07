// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (wasm && wasm.stackswitch) || (wasm32 && wasm32.stackswitch)

package runtime

var WasmContTableSize = wasmContTableSize
var WasmContTableGrow = wasmContTableGrow

func WasmContSlot() int32 {
	return getg().wasmCont.index
}
