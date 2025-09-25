// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cryptotest

import (
	"crypto/internal/fips140"
	"testing"
)

func MustSupportFIPS140(t *testing.T) {
	t.Helper()
	if err := fips140.Supported(); err != nil {
		t.Skipf("test requires FIPS 140 mode: %v", err)
	}
}
