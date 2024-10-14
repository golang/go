// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ecdsa_test

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"testing"
)

func testValid(t *testing.T, c elliptic.Curve) {
	private, _ := ecdsa.GenerateKey(c, rand.Reader)
	toobigpub := private.PublicKey

	//make Y greater than P
	toobigpub.Y.Add(toobigpub.Y, c.Params().P)

	//point is on the curve
	if !c.IsOnCurve(toobigpub.X, toobigpub.Y) {
		t.Error("test point should be on curve")
	}

	//.. but is not valid because Y is greater than P
	if toobigpub.IsValid() {
		t.Error("public key is on the curve but contains coordinate greater than the curve's prime")
	}

}

func TestValid(t *testing.T) {
	t.Run("P256", func(t *testing.T) { testValid(t, elliptic.P256()) })
	if testing.Short() {
		return
	}
	t.Run("P384", func(t *testing.T) { testValid(t, elliptic.P384()) })
	t.Run("P521", func(t *testing.T) { testValid(t, elliptic.P521()) })
}
