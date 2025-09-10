// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cryptotest

import (
	"crypto/internal/boring"
	"crypto/internal/fips140"
	"hash"
	"internal/testhash"
	"io"
	"math/rand"
	"testing"
	"time"
)

type MakeHash func() hash.Hash

// TestHash performs a set of tests on hash.Hash implementations, checking the
// documented requirements of Write, Sum, Reset, Size, and BlockSize.
func TestHash(t *testing.T, mh MakeHash) {
	if boring.Enabled || fips140.Version() == "v1.0.0" {
		testhash.TestHashWithoutClone(t, testhash.MakeHash(mh))
		return
	}
	testhash.TestHash(t, testhash.MakeHash(mh))
}

func newRandReader(t *testing.T) io.Reader {
	seed := time.Now().UnixNano()
	t.Logf("Deterministic RNG seed: 0x%x", seed)
	return rand.New(rand.NewSource(seed))
}
