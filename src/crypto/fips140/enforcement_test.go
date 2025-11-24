// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fips140_test

import (
	"crypto/internal/cryptotest"
	"internal/testenv"
	"path/filepath"
	"strings"
	"testing"
)

func TestWithoutEnforcement(t *testing.T) {
	testenv.MustHaveExec(t)
	testenv.MustHaveGoBuild(t)
	cryptotest.MustSupportFIPS140(t)

	tool, _ := testenv.GoTool()
	tmpdir := t.TempDir()
	binFile := filepath.Join(tmpdir, "fips140.test")
	cmd := testenv.Command(t, tool, "test", "-c", "-o", binFile, "./testdata")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Log(string(out))
		t.Errorf("Could not build enforcement tests")
	}
	cmd = testenv.Command(t, binFile, "-test.list", ".")
	list, err := cmd.CombinedOutput()
	if err != nil {
		t.Log(string(out))
		t.Errorf("Could not get enforcement test list")
	}
	for test := range strings.Lines(string(list)) {
		test = strings.TrimSpace(test)
		t.Run(test, func(t *testing.T) {
			cmd = testenv.Command(t, binFile, "-test.run", "^"+test+"$")
			cmd.Env = append(cmd.Env, "GODEBUG=fips140=only")
			out, err := cmd.CombinedOutput()
			if err != nil {
				t.Error(string(out))
			}
		})
	}
}
