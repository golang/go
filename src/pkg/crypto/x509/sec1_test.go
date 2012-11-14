// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x509

import (
	"encoding/hex"
	"testing"
)

// Generated using:
//   openssl ecparam -genkey -name secp384r1 -outform PEM
var ecPrivateKeyHex = `3081a40201010430bdb9839c08ee793d1157886a7a758a3c8b2a17a4df48f17ace57c72c56b4723cf21dcda21d4e1ad57ff034f19fcfd98ea00706052b81040022a16403620004feea808b5ee2429cfcce13c32160e1c960990bd050bb0fdf7222f3decd0a55008e32a6aa3c9062051c4cba92a7a3b178b24567412d43cdd2f882fa5addddd726fe3e208d2c26d733a773a597abb749714df7256ead5105fa6e7b3650de236b50`

func TestParseECPrivateKey(t *testing.T) {
	derBytes, _ := hex.DecodeString(ecPrivateKeyHex)
	_, err := ParseECPrivateKey(derBytes)
	if err != nil {
		t.Errorf("failed to decode EC private key: %s", err)
	}
}
