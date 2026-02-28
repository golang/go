// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"internal/testenv"
	"path/filepath"
	"testing"
)

func TestNooptCgoBuild(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode.")
	}
	t.Parallel()

	testenv.MustHaveGoBuild(t)
	testenv.MustHaveCGO(t)
	dir := t.TempDir()
	cmd := testenv.Command(t, testenv.GoToolPath(t), "build", "-gcflags=-N -l", "-o", filepath.Join(dir, "a.out"))
	cmd.Dir = filepath.Join(testenv.GOROOT(t), "src", "runtime", "testdata", "testprogcgo")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Logf("go build output: %s", out)
		t.Fatal(err)
	}
}
