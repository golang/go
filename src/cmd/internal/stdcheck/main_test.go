// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
package main

import (
	"bytes"
	"internal/testenv"
	"os"
	"os/exec"
	"testing"
)

func TestMain(m *testing.M) {
	if os.Getenv("GO_STDCHECKTEST_IS_STDCHECK") != "" {
		main()
		os.Exit(0)
	}

	os.Exit(m.Run())
}

func TestRunAnalyzers(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in short mode")
	}
	cmd := exec.Command(testenv.GoToolPath(t), "vet", "-vettool", testenv.Executable(t), "std")
	cmd.Env = append(os.Environ(), "GO_STDCHECKTEST_IS_STDCHECK=1")
	stderr := new(bytes.Buffer)
	cmd.Stderr = stderr
	if err := cmd.Run(); err != nil {
		// TODO: we could probably do a better job of parsing and presenting the
		// output here (e.g. one subtest per analyzer?)
		t.Error(stderr.String())
	}
}
