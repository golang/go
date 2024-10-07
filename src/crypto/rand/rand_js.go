// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rand

// The maximum buffer size for crypto.getRandomValues is 65536 bytes.
// https://developer.mozilla.org/en-US/docs/Web/API/Crypto/getRandomValues#exceptions
const maxGetRandomRead = 64 << 10

//go:wasmimport gojs runtime.getRandomData
//go:noescape
func getRandomValues(r []byte)

// read calls the JavaScript Crypto.getRandomValues() method.
// See https://developer.mozilla.org/en-US/docs/Web/API/Crypto/getRandomValues.
func read(b []byte) error {
	for len(b) > 0 {
		size := len(b)
		if size > maxGetRandomRead {
			size = maxGetRandomRead
		}
		getRandomValues(b[:size])
		b = b[size:]
	}
	return nil
}
