// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build s390x && !purego

package ecdsa

import (
	"crypto/elliptic"
	"testing"
)

func TestNoAsm(t *testing.T) {
	testingDisableKDSA = true
	defer func() { testingDisableKDSA = false }()

	curves := [...]elliptic.Curve{
		elliptic.P256(),
		elliptic.P384(),
		elliptic.P521(),
	}

	for _, curve := range curves {
		name := curve.Params().Name
		t.Run(name, func { t -> testKeyGeneration(t, curve) })
		t.Run(name, func { t -> testSignAndVerify(t, curve) })
		t.Run(name, func { t -> testNonceSafety(t, curve) })
		t.Run(name, func { t -> testINDCCA(t, curve) })
		t.Run(name, func { t -> testNegativeInputs(t, curve) })
	}
}
