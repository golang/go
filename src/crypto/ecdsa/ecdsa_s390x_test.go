// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build s390x

package ecdsa

import (
	"crypto/elliptic"
	"testing"
)

func TestNoAsm(t *testing.T) {
	curves := [...]elliptic.Curve{
		elliptic.P256(),
		elliptic.P384(),
		elliptic.P521(),
	}

	for _, curve := range curves {
		// override the name of the curve to stop the assembly path being taken
		params := *curve.Params()
		name := params.Name
		params.Name = name + "_GENERIC_OVERRIDE"

		testKeyGeneration(t, &params, name)
		testSignAndVerify(t, &params, name)
		testNonceSafety(t, &params, name)
		testINDCCA(t, &params, name)
		testNegativeInputs(t, &params, name)
	}
}
