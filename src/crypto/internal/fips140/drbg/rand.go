// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package drbg provides cryptographically secure random bytes
// usable by FIPS code. In FIPS mode it uses an SP 800-90A Rev. 1
// Deterministic Random Bit Generator (DRBG). Otherwise,
// it uses the operating system's random number generator.
package drbg

import (
	"crypto/internal/fips140"
	"crypto/internal/sysrand"
	"io"
)

// Read fills b with cryptographically secure random bytes. In FIPS mode, it
// uses an SP 800-90A Rev. 1 Deterministic Random Bit Generator (DRBG).
// Otherwise, it uses the operating system's random number generator.
func Read(b []byte) {
	if testingReader != nil {
		fips140.RecordNonApproved()
		// Avoid letting b escape in the non-testing case.
		bb := make([]byte, len(b))
		testingReader.Read(bb)
		copy(b, bb)
		return
	}

	if !fips140.Enabled {
		sysrand.Read(b)
		return
	}

	readFromEntropy(b)
}

var testingReader io.Reader

// SetTestingReader sets a global, deterministic cryptographic randomness source
// for testing purposes. Its Read method must never return an error, it must
// never return short, and it must be safe for concurrent use.
//
// This is only intended to be used by the testing/cryptotest package.
func SetTestingReader(r io.Reader) {
	testingReader = r
}

// DefaultReader is a sentinel type, embedded in the default
// [crypto/rand.Reader], used to recognize it when passed to
// APIs that accept a rand io.Reader.
//
// Any [io.Reader] that embeds this type is assumed to
// call [Read] as its [io.Reader.Read] method.
type DefaultReader struct{}

func (d DefaultReader) defaultReader() {}

// IsDefaultReader reports whether the r embeds the [DefaultReader] type.
func IsDefaultReader(r io.Reader) bool {
	_, ok := r.(interface{ defaultReader() })
	return ok
}

// ReadWithReader uses Reader to fill b with cryptographically secure random
// bytes. It is intended for use in APIs that expose a rand io.Reader.
func ReadWithReader(r io.Reader, b []byte) error {
	if IsDefaultReader(r) {
		Read(b)
		return nil
	}

	fips140.RecordNonApproved()
	_, err := io.ReadFull(r, b)
	return err
}
