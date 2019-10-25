// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hash_test

import (
	"bytes/hash"
	basehash "hash"
	"testing"
)

func TestUnseededHash(t *testing.T) {
	m := map[uint64]struct{}{}
	for i := 0; i < 1000; i++ {
		h := hash.New()
		m[h.Hash()] = struct{}{}
	}
	if len(m) < 900 {
		t.Errorf("empty hash not sufficiently random: got %d, want 1000", len(m))
	}
}

func TestSeededHash(t *testing.T) {
	s := hash.MakeSeed(1234)
	m := map[uint64]struct{}{}
	for i := 0; i < 1000; i++ {
		h := hash.New()
		h.SetSeed(s)
		m[h.Hash()] = struct{}{}
	}
	if len(m) != 1 {
		t.Errorf("seeded hash is random: got %d, want 1", len(m))
	}
}

func TestHashGrouping(t *testing.T) {
	b := []byte("foo")
	h1 := hash.New()
	h2 := hash.New()
	h2.SetSeed(h1.Seed())
	h1.AddBytes(b)
	for _, x := range b {
		h2.AddByte(x)
	}
	if h1.Hash() != h2.Hash() {
		t.Errorf("hash of \"foo\" and \"f\",\"o\",\"o\" not identical")
	}
}

func TestHashBytesVsString(t *testing.T) {
	s := "foo"
	b := []byte(s)
	h1 := hash.New()
	h2 := hash.New()
	h2.SetSeed(h1.Seed())
	h1.AddString(s)
	h2.AddBytes(b)
	if h1.Hash() != h2.Hash() {
		t.Errorf("hash of string and byts not identical")
	}
}

func TestHashHighBytes(t *testing.T) {
	// See issue 34925.
	const N = 10
	m := map[uint64]struct{}{}
	for i := 0; i < N; i++ {
		h := hash.New()
		h.AddString("foo")
		m[h.Hash()>>32] = struct{}{}
	}
	if len(m) < N/2 {
		t.Errorf("from %d seeds, wanted at least %d different hashes; got %d", N, N/2, len(m))
	}
}

// Make sure a Hash implements the hash.Hash and hash.Hash64 interfaces.
var _ basehash.Hash = &hash.Hash{}
var _ basehash.Hash64 = &hash.Hash{}
