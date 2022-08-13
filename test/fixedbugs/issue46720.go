// compile

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func f() {
	nonce := make([]byte, 24)
	g((*[24]byte)(nonce))
}

//go:noinline
func g(*[24]byte) {}
