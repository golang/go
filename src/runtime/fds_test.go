// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

package runtime_test

import (
	"internal/testenv"
	"os"
	"strings"
	"testing"
)

func TestCheckFDs(t *testing.T) {
	if *flagQuick {
		t.Skip("-quick")
	}

	testenv.MustHaveGoBuild(t)

	fdsBin, err := buildTestProg(t, "testfds")
	if err != nil {
		t.Fatal(err)
	}

	i, err := os.CreateTemp(t.TempDir(), "fds-input")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := i.Write([]byte("stdin")); err != nil {
		t.Fatal(err)
	}
	if err := i.Close(); err != nil {
		t.Fatal(err)
	}

	o, err := os.CreateTemp(t.TempDir(), "fds-output")
	if err != nil {
		t.Fatal(err)
	}
	outputPath := o.Name()
	if err := o.Close(); err != nil {
		t.Fatal(err)
	}

	env := []string{"TEST_OUTPUT=" + outputPath}
	for _, e := range os.Environ() {
		if strings.HasPrefix(e, "GODEBUG=") || strings.HasPrefix(e, "GOTRACEBACK=") {
			continue
		}
		env = append(env, e)
	}

	proc, err := os.StartProcess(fdsBin, []string{fdsBin}, &os.ProcAttr{
		Env:   env,
		Files: []*os.File{},
	})
	if err != nil {
		t.Fatal(err)
	}
	ps, err := proc.Wait()
	if err != nil {
		t.Fatal(err)
	}
	if ps.ExitCode() != 0 {
		t.Fatalf("testfds failed: %d", ps.ExitCode())
	}

	fc, err := os.ReadFile(outputPath)
	if err != nil {
		t.Fatal(err)
	}
	if string(fc) != "" {
		t.Errorf("unexpected file content, got: %q", string(fc))
	}
}
