// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build amd64 arm64 ppc64le

package elliptic

import (
	"crypto/rand"
	"testing"
	"time"
)

func TestFuzz(t *testing.T) {

	p256 := P256()
	p256Generic := p256.Params()

	var scalar1 [32]byte
	var scalar2 [32]byte
	var timeout *time.Timer

	if testing.Short() {
		timeout = time.NewTimer(10 * time.Millisecond)
	} else {
		timeout = time.NewTimer(2 * time.Second)
	}

	for {
		select {
		case <-timeout.C:
			return
		default:
		}

		rand.Read(scalar1[:])
		rand.Read(scalar2[:])

		x, y := p256.ScalarBaseMult(scalar1[:])
		x2, y2 := p256Generic.ScalarBaseMult(scalar1[:])

		xx, yy := p256.ScalarMult(x, y, scalar2[:])
		xx2, yy2 := p256Generic.ScalarMult(x2, y2, scalar2[:])

		if x.Cmp(x2) != 0 || y.Cmp(y2) != 0 {
			t.Fatalf("ScalarBaseMult does not match reference result with scalar: %x, please report this error to security@golang.org", scalar1)
		}

		if xx.Cmp(xx2) != 0 || yy.Cmp(yy2) != 0 {
			t.Fatalf("ScalarMult does not match reference result with scalars: %x and %x, please report this error to security@golang.org", scalar1, scalar2)
		}
	}
}
