// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cryptotest

import (
	"crypto/internal/fips140"
	"internal/testenv"
	"regexp"
	"testing"
)

func MustSupportFIPS140(t *testing.T) {
	t.Helper()
	if err := fips140.Supported(); err != nil {
		t.Skipf("test requires FIPS 140 mode: %v", err)
	}
}

func RerunWithFIPS140Enabled(t *testing.T) {
	t.Helper()
	MustSupportFIPS140(t)
	nameRegex := "^" + regexp.QuoteMeta(t.Name()) + "$"
	cmd := testenv.Command(t, testenv.Executable(t), "-test.run="+nameRegex, "-test.v")
	cmd.Env = append(cmd.Environ(), "GODEBUG=fips140=on")
	out, err := cmd.CombinedOutput()
	t.Logf("running with GODEBUG=fips140=on:\n%s", out)
	if err != nil {
		t.Errorf("fips140=on subprocess failed: %v", err)
	}
}

func RerunWithFIPS140Enforced(t *testing.T) {
	t.Helper()
	MustSupportFIPS140(t)
	nameRegex := "^" + regexp.QuoteMeta(t.Name()) + "$"
	cmd := testenv.Command(t, testenv.Executable(t), "-test.run="+nameRegex, "-test.v")
	cmd.Env = append(cmd.Environ(), "GODEBUG=fips140=only")
	out, err := cmd.CombinedOutput()
	t.Logf("running with GODEBUG=fips140=only:\n%s", out)
	if err != nil {
		t.Errorf("fips140=only subprocess failed: %v", err)
	}
}
