// errorcheck -0 -m

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import (
	"crypto/ecdh"
	"crypto/rand"
)

func F(peerShare []byte) ([]byte, error) { // ERROR "leaking param: peerShare"
	p256 := ecdh.P256() // ERROR "inlining call to ecdh.P256"

	ourKey, err := p256.GenerateKey(rand.Reader) // ERROR "devirtualizing p256.GenerateKey" "inlining call to ecdh.*GenerateKey"
	if err != nil {
		return nil, err
	}

	peerPublic, err := p256.NewPublicKey(peerShare) // ERROR "devirtualizing p256.NewPublicKey" "inlining call to ecdh.*NewPublicKey"
	if err != nil {
		return nil, err
	}

	return ourKey.ECDH(peerPublic)
}

// Test that inlining doesn't break if devirtualization exposes a new
// inlinable callee.

func f() { // ERROR "can inline f"
	var i interface{ m() } = T(0) // ERROR "T\(0\) does not escape"
	i.m()                         // ERROR "devirtualizing i.m" "inlining call to T.m"
}

type T int

func (T) m() { // ERROR "can inline T.m"
	if never {
		f() // ERROR "inlining call to f" "devirtualizing i.m" "T\(0\) does not escape"
	}
}

var never bool
