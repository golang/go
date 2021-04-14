// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"bytes"
	"internal/testenv"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func TestDeadcode(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	t.Parallel()

	tmpdir, err := ioutil.TempDir("", "TestDeadcode")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)

	tests := []struct {
		src     string
		pattern string
	}{
		{"reflectcall", "main.T.M"},
		{"typedesc", "type.main.T"},
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
			if bytes.Contains(out, []byte(test.pattern)) {
				t.Errorf("%s should not be reachable. Output:\n%s", test.pattern, out)
			}
		})
	}
}
