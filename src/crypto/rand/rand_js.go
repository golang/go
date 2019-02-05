// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build js,wasm

package rand

import "syscall/js"

func init() {
	Reader = &reader{}
}

var jsCrypto = js.Global().Get("crypto")

// reader implements a pseudorandom generator
// using JavaScript crypto.getRandomValues method.
// See https://developer.mozilla.org/en-US/docs/Web/API/Crypto/getRandomValues.
type reader struct{}

func (r *reader) Read(b []byte) (int, error) {
	a := js.TypedArrayOf(b)
	jsCrypto.Call("getRandomValues", a)
	a.Release()
	return len(b), nil
}
