// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rsa

import (
	"crypto"
	"crypto/rand"
	"encoding/asn1"
	"reflect"
	"testing"
	"unsafe"
)

func TestBoringASN1Marshal(t *testing.T) {
	k, err := GenerateKey(rand.Reader, 128)
	if err != nil {
		t.Fatal(err)
	}
	// This used to fail, because of the unexported 'boring' field.
	// Now the compiler hides it [sic].
	_, err = asn1.Marshal(k.PublicKey)
	if err != nil {
		t.Fatal(err)
	}
}

func TestBoringDeepEqual(t *testing.T) {
	k, err := GenerateKey(rand.Reader, 128)
	if err != nil {
		t.Fatal(err)
	}
	k.boring = nil // probably nil already but just in case
	k2 := *k
	k2.boring = unsafe.Pointer(k) // anything not nil, for this test
	if !reflect.DeepEqual(k, &k2) {
		// compiler should be hiding the boring field from reflection
		t.Fatalf("DeepEqual compared boring fields")
	}
}

func TestBoringVerify(t *testing.T) {
	// This changed behavior and broke golang.org/x/crypto/openpgp.
	// Go accepts signatures without leading 0 padding, while BoringCrypto does not.
	// So the Go wrappers must adapt.
	key := &PublicKey{
		N: bigFromHex("c4fdf7b40a5477f206e6ee278eaef888ca73bf9128a9eef9f2f1ddb8b7b71a4c07cfa241f028a04edb405e4d916c61d6beabc333813dc7b484d2b3c52ee233c6a79b1eea4e9cc51596ba9cd5ac5aeb9df62d86ea051055b79d03f8a4fa9f38386f5bd17529138f3325d46801514ea9047977e0829ed728e68636802796801be1"),
		E: 65537,
	}

	hash := fromHex("019c5571724fb5d0e47a4260c940e9803ba05a44")
	paddedHash := fromHex("3021300906052b0e03021a05000414019c5571724fb5d0e47a4260c940e9803ba05a44")

	// signature is one byte shorter than key.N.
	sig := fromHex("5edfbeb6a73e7225ad3cc52724e2872e04260d7daf0d693c170d8c4b243b8767bc7785763533febc62ec2600c30603c433c095453ede59ff2fcabeb84ce32e0ed9d5cf15ffcbc816202b64370d4d77c1e9077d74e94a16fb4fa2e5bec23a56d7a73cf275f91691ae1801a976fcde09e981a2f6327ac27ea1fecf3185df0d56")

	err := VerifyPKCS1v15(key, 0, paddedHash, sig)
	if err != nil {
		t.Errorf("raw: %v", err)
	}

	err = VerifyPKCS1v15(key, crypto.SHA1, hash, sig)
	if err != nil {
		t.Errorf("sha1: %v", err)
	}
}
