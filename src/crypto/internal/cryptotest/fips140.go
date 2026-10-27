// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cryptotest

import (
	"crypto/internal/fips140"
	"strconv"
	"strings"
	"testing"
)

func MustSupportFIPS140(t *testing.T) {
	t.Helper()
	if err := fips140.Supported(); err != nil {
		t.Skipf("test requires FIPS 140 mode: %v", err)
	}
}

// MustMinimumFIPS140ModuleVersion skips the test if compiled against a lower
// minor version of the FIPS 140-3 module than min (such as "v1.26.0").
func MustMinimumFIPS140ModuleVersion(tb testing.TB, min string) {
	tb.Helper()
	if fips140.Version() == "latest" {
		return
	}
	if parseFIPS140MinorVersion(tb, fips140.Version()) < parseFIPS140MinorVersion(tb, min) {
		tb.Skipf("test requires FIPS 140-3 module %s or later", min)
	}
}

func parseFIPS140MinorVersion(tb testing.TB, version string) int {
	tb.Helper()
	v, ok := strings.CutPrefix(version, "v1.")
	if !ok {
		tb.Fatalf("unexpected FIPS 140 version format: %q", version)
	}
	v, _, ok = strings.Cut(v, ".")
	if !ok {
		tb.Fatalf("unexpected FIPS 140 version format: %q", version)
	}
	i, err := strconv.Atoi(v)
	if err != nil {
		tb.Fatalf("unexpected FIPS 140 version format %q: %v", version, err)
	}
	return i
}
