// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rt

// errmsg is the text of the last error that WasmError recorded. On wasm,
// _logos_errmsg_ptr and _logos_errmsg_len export its address and length so
// the host loader can read it from linear memory.
var errmsg []byte

// WasmError is the error exit of a //go:wasmexport or //logos:wasmexport
// function (SPEC Sec 14.2). A nil err returns 0, the errCode of success. A
// non-nil err records err.Error() as the last error message and returns 1,
// so the loader wrapper throws that text on the host side. The message
// stays readable until the next non-nil call.
func WasmError(err error) int32 {
	if err == nil {
		return 0
	}
	errmsg = []byte(err.Error())
	return 1
}
