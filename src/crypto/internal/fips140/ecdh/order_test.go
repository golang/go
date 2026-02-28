// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ecdh

import (
	"bytes"
	"crypto/elliptic"
	"testing"
)

func TestOrders(t *testing.T) {
	if !bytes.Equal(elliptic.P224().Params().N.Bytes(), P224().N) {
		t.Errorf("P-224 order mismatch")
	}
	if !bytes.Equal(elliptic.P256().Params().N.Bytes(), P256().N) {
		t.Errorf("P-256 order mismatch")
	}
	if !bytes.Equal(elliptic.P384().Params().N.Bytes(), P384().N) {
		t.Errorf("P-384 order mismatch")
	}
	if !bytes.Equal(elliptic.P521().Params().N.Bytes(), P521().N) {
		t.Errorf("P-521 order mismatch")
	}
}
