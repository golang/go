// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package cryptotest provides deterministic random source testing.
package cryptotest

import (
	cryptorand "crypto/rand"
	"internal/byteorder"
	"io"
	mathrand "math/rand/v2"
	"sync"
	"testing"

	// Import unsafe and crypto/rand, which imports crypto/internal/rand,
	// for the crypto/internal/rand.SetTestingReader go:linkname.
	_ "crypto/rand"
	_ "unsafe"
)

//go:linkname randSetTestingReader crypto/internal/rand.SetTestingReader
func randSetTestingReader(r io.Reader)

//go:linkname testingCheckParallel testing.checkParallel
func testingCheckParallel(t *testing.T)

// SetGlobalRandom sets a global, deterministic cryptographic randomness source
// for the duration of test t. It affects crypto/rand, and all implicit sources
// of cryptographic randomness in the crypto/... packages.
//
// SetGlobalRandom may be called multiple times in the same test to reset the
// random stream or change the seed.
//
// Because SetGlobalRandom affects the whole process, it cannot be used in
// parallel tests or tests with parallel ancestors.
//
// Note that the way cryptographic algorithms use randomness is generally not
// specified and may change over time. Thus, if a test expects a specific output
// from a cryptographic function, it may fail in the future even if it uses
// SetGlobalRandom.
//
// SetGlobalRandom is not supported when building against the Go Cryptographic
// Module v1.0.0 (i.e. when [crypto/fips140.Version] returns "v1.0.0").
func SetGlobalRandom(t *testing.T, seed uint64) {
	if t == nil {
		panic("cryptotest: SetGlobalRandom called with a nil *testing.T")
	}
	if !testing.Testing() {
		panic("cryptotest: SetGlobalRandom used in a non-test binary")
	}
	testingCheckParallel(t)

	var s [32]byte
	byteorder.LEPutUint64(s[:8], seed)
	r := &lockedReader{r: mathrand.NewChaCha8(s)}

	randSetTestingReader(r)
	previous := cryptorand.Reader
	cryptorand.Reader = r
	t.Cleanup(func() {
		cryptorand.Reader = previous
		randSetTestingReader(nil)
	})
}

type lockedReader struct {
	sync.Mutex
	r *mathrand.ChaCha8
}

func (lr *lockedReader) Read(b []byte) (n int, err error) {
	lr.Lock()
	defer lr.Unlock()
	return lr.r.Read(b)
}
