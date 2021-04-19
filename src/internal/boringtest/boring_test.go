// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Like crypto/rsa/boring_test.go but outside the crypto/ tree.
// Tests what happens if a package outside the crypto/ tree
// "adopts" a struct definition. This happens in golang.org/x/crypto/ssh.

package boring

import (
	"crypto/rand"
	"crypto/rsa"
	"encoding/asn1"
	"reflect"
	"testing"
)

type publicKey rsa.PublicKey

func TestBoringASN1Marshal(t *testing.T) {
	k, err := rsa.GenerateKey(rand.Reader, 128)
	if err != nil {
		t.Fatal(err)
	}
	pk := (*publicKey)(&k.PublicKey)
	// This used to fail, because of the unexported 'boring' field.
	// Now the compiler hides it [sic].
	_, err = asn1.Marshal(*pk)
	if err != nil {
		t.Fatal(err)
	}
}

func TestBoringDeepEqual(t *testing.T) {
	k0, err := rsa.GenerateKey(rand.Reader, 128)
	if err != nil {
		t.Fatal(err)
	}
	k := (*publicKey)(&k0.PublicKey)
	k2 := *k
	rsa.EncryptPKCS1v15(rand.Reader, (*rsa.PublicKey)(&k2), []byte("hello")) // initialize hidden boring field
	if !reflect.DeepEqual(k, &k2) {
		// compiler should be hiding the boring field from reflection
		t.Fatalf("DeepEqual compared boring fields")
	}
}
