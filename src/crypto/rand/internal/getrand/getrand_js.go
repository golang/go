// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build js && wasm

package getrand

import (
	"syscall/js"
)

// The maximum buffer size for crypto.getRandomValues is 65536 bytes.
// https://developer.mozilla.org/en-US/docs/Web/API/Crypto/getRandomValues#exceptions
const maxGetRandomRead = 64 << 10

var jsCrypto = js.Global().Get("crypto")
var uint8Array = js.Global().Get("Uint8Array")

// getRandom populates the input slice with pseudorandom data
// using JavaScript crypto.getRandomValues method.
// See https://developer.mozilla.org/en-US/docs/Web/API/Crypto/getRandomValues.
func getRandom(b []byte) error {
	a := uint8Array.New(len(b))
	jsCrypto.Call("getRandomValues", a)
	js.CopyBytesToGo(b, a)
	return nil
}
