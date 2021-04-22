// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"bytes"
	"internal/testenv"
	"os/exec"
	"path/filepath"
	"testing"
)

func TestDeadcode(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	t.Parallel()

	tmpdir := t.TempDir()

	tests := []struct {
		src      string
		pos, neg string // positive and negative patterns
	}{
		{"reflectcall", "", "main.T.M"},
		{"typedesc", "", "type.main.T"},
		{"ifacemethod", "", "main.T.M"},
		{"ifacemethod2", "main.T.M", ""},
		{"ifacemethod3", "main.S.M", ""},
		{"ifacemethod4", "", "main.T.M"},
	}
	for _, test := range tests {
		test := test
		t.Run(test.src, func(t *testing.T) {
			t.Parallel()
			src := filepath.Join("testdata", "deadcode", test.src+".go")
			exe := filepath.Join(tmpdir, test.src+".exe")
			cmd := exec.Command(testenv.GoToolPath(t), "build", "-ldflags=-dumpdep", "-o", exe, src)
			out, err := cmd.CombinedOutput()
			if err != nil {
				t.Fatalf("%v: %v:\n%s", cmd.Args, err, out)
			}
			if test.pos != "" && !bytes.Contains(out, []byte(test.pos+"\n")) {
				t.Errorf("%s should be reachable. Output:\n%s", test.pos, out)
			}
			if test.neg != "" && bytes.Contains(out, []byte(test.neg+"\n")) {
				t.Errorf("%s should not be reachable. Output:\n%s", test.neg, out)
			}
		})
	}
}
