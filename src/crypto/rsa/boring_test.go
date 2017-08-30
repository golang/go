// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rsa

import (
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
