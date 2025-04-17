// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ecdsa

import (
	"bytes"
	"crypto/internal/fips140/bigmod"
	"crypto/rand"
	"io"
	"testing"
)

func TestRandomPoint(t *testing.T) {
	t.Run("P-224", func(t *testing.T) { testRandomPoint(t, P224()) })
	t.Run("P-256", func(t *testing.T) { testRandomPoint(t, P256()) })
	t.Run("P-384", func(t *testing.T) { testRandomPoint(t, P384()) })
	t.Run("P-521", func(t *testing.T) { testRandomPoint(t, P521()) })
}

func testRandomPoint[P Point[P]](t *testing.T, c *Curve[P]) {
	t.Cleanup(func() { testingOnlyRejectionSamplingLooped = nil })
	var loopCount int
	testingOnlyRejectionSamplingLooped = func() { loopCount++ }

	// A sequence of all ones will generate 2^N-1, which should be rejected.
	// (Unless, for example, we are masking too many bits.)
	r := io.MultiReader(bytes.NewReader(bytes.Repeat([]byte{0xff}, 100)), rand.Reader)
	if k, p, err := randomPoint(c, func(b []byte) error {
		_, err := r.Read(b)
		return err
	}); err != nil {
		t.Fatal(err)
	} else if k.IsZero() == 1 {
		t.Error("k is zero")
	} else if p.Bytes()[0] != 4 {
		t.Error("p is infinity")
	}
	if loopCount == 0 {
		t.Error("overflow was not rejected")
	}
	loopCount = 0

	// A sequence of all zeroes will generate zero, which should be rejected.
	r = io.MultiReader(bytes.NewReader(bytes.Repeat([]byte{0}, 100)), rand.Reader)
	if k, p, err := randomPoint(c, func(b []byte) error {
		_, err := r.Read(b)
		return err
	}); err != nil {
		t.Fatal(err)
	} else if k.IsZero() == 1 {
		t.Error("k is zero")
	} else if p.Bytes()[0] != 4 {
		t.Error("p is infinity")
	}
	if loopCount == 0 {
		t.Error("zero was not rejected")
	}
	loopCount = 0

	// P-256 has a 2⁻³² chance of randomly hitting a rejection. For P-224 it's
	// 2⁻¹¹², for P-384 it's 2⁻¹⁹⁴, and for P-521 it's 2⁻²⁶², so if we hit in
	// tests, something is horribly wrong. (For example, we are masking the
	// wrong bits.)
	if c.curve == p256 {
		return
	}
	if k, p, err := randomPoint(c, func(b []byte) error {
		_, err := rand.Reader.Read(b)
		return err
	}); err != nil {
		t.Fatal(err)
	} else if k.IsZero() == 1 {
		t.Error("k is zero")
	} else if p.Bytes()[0] != 4 {
		t.Error("p is infinity")
	}
	if loopCount > 0 {
		t.Error("unexpected rejection")
	}
}

func TestHashToNat(t *testing.T) {
	t.Run("P-224", func(t *testing.T) { testHashToNat(t, P224()) })
	t.Run("P-256", func(t *testing.T) { testHashToNat(t, P256()) })
	t.Run("P-384", func(t *testing.T) { testHashToNat(t, P384()) })
	t.Run("P-521", func(t *testing.T) { testHashToNat(t, P521()) })
}

func testHashToNat[P Point[P]](t *testing.T, c *Curve[P]) {
	for l := 0; l < 600; l++ {
		h := bytes.Repeat([]byte{0xff}, l)
		hashToNat(c, bigmod.NewNat(), h)
	}
}
