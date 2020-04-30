// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !s390x

package ed25519

func sign(signature, privateKey, message []byte) {
	signGeneric(signature, privateKey, message)
}

func verify(publicKey PublicKey, message, sig []byte) bool {
	return verifyGeneric(publicKey, message, sig)
}
