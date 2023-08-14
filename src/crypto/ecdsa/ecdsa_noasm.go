// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !s390x

package ecdsa

import "io"

func verifyAsm(pub *PublicKey, hash []byte, sig []byte) error {
	return errNoAsm
}

func signAsm(priv *PrivateKey, csprng io.Reader, hash []byte) (sig []byte, err error) {
	return nil, errNoAsm
}
