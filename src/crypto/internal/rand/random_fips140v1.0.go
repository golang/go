// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build fips140v1.0 || fips140v1.26

package rand

import (
	"crypto/internal/fips140/drbg"
	"io"
)

// IsDefaultReader reports whether r is the default [crypto/rand.Reader].
//
// If true, the Read method of r can be assumed to call [drbg.Read].
func IsDefaultReader(r io.Reader) bool {
	_, ok := r.(drbg.DefaultReader)
	return ok
}
