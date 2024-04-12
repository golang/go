// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build js && wasm

package rand

import "syscall/js"

var batchedGetRandom func([]byte) error

var randReader = reader{}

func init() {
	batchedGetRandom = batched(getRandom, maxGetRandomRead)
}

var jsCrypto = js.Global().Get("crypto")
var uint8Array = js.Global().Get("Uint8Array")

// reader implements a pseudorandom generator
// using JavaScript crypto.getRandomValues method.
// See https://developer.mozilla.org/en-US/docs/Web/API/Crypto/getRandomValues.
type reader struct{}

func (r reader) Read(b []byte) (int, error) {
	if err := batchedGetRandom(b); err != nil {
		return 0, err
	}
	return len(b), nil
}

func getRandom(b []byte) error {
	a := uint8Array.New(len(b))
	jsCrypto.Call("getRandomValues", a)
	js.CopyBytesToGo(b, a)
	return nil
}
