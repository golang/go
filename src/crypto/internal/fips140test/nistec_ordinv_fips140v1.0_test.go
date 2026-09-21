// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build fips140v1.0 || fips140v1.26

package fipstest

import (
	"crypto/internal/fips140/nistec"
	"internal/goarch"
	"testing"
)

// package nistec
// func P256OrdInverse(k []byte) ([]byte, error)

func p256OrdInverse(t *testing.T, k *[4]uint64) {
	input := limbsToBytes(*k)
	out, err := nistec.P256OrdInverse(input)
	if err != nil {
		switch goarch.GOARCH {
		case "amd64", "arm64":
			t.Fatal(err)
		default:
			t.Skip("this GOARCH didn't have P256OrdInverse in v1.0/v1.26")
		}
	}
	*k = bytesToLimbs(out)
}
