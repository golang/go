// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os_test

import (
	"fmt"
	"internal/testenv"
	"os"
	osexec "os/exec"
	"path/filepath"
	"runtime"
	"testing"
)

const executable_EnvVar = "OSTEST_OUTPUT_EXECPATH"

func TestExecutable(t *testing.T) {
	testenv.MustHaveExec(t) // will also execlude nacl, which doesn't support Executable anyway
	ep, err := os.Executable()
	if err != nil {
		switch goos := runtime.GOOS; goos {
		case "openbsd": // procfs is not mounted by default
			t.Skipf("Executable failed on %s: %v, expected", goos, err)
		}
		t.Fatalf("Executable failed: %v", err)
	}
	// we want fn to be of the form "dir/prog"
	dir := filepath.Dir(filepath.Dir(ep))
	fn, err := filepath.Rel(dir, ep)
	if err != nil {
		t.Fatalf("filepath.Rel: %v", err)
	}
	cmd := &osexec.Cmd{}
	// make child start with a relative program path
	cmd.Dir = dir
	cmd.Path = fn
	// forge argv[0] for child, so that we can verify we could correctly
	// get real path of the executable without influenced by argv[0].
	cmd.Args = []string{"-", "-test.run=XXXX"}
	cmd.Env = append(os.Environ(), fmt.Sprintf("%s=1", executable_EnvVar))
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("exec(self) failed: %v", err)
	}
	outs := string(out)
	if !filepath.IsAbs(outs) {
		t.Fatalf("Child returned %q, want an absolute path", out)
	}
	if !sameFile(outs, ep) {
		t.Fatalf("Child returned %q, not the same file as %q", out, ep)
	}
}

func sameFile(fn1, fn2 string) bool {
	fi1, err := os.Stat(fn1)
	if err != nil {
		return false
	}
	fi2, err := os.Stat(fn2)
	if err != nil {
		return false
	}
	return os.SameFile(fi1, fi2)
}

func init() {
	if e := os.Getenv(executable_EnvVar); e != "" {
		// first chdir to another path
		dir := "/"
		if runtime.GOOS == "windows" {
			cwd, err := os.Getwd()
			if err != nil {
				panic(err)
			}
			dir = filepath.VolumeName(cwd)
		}
		os.Chdir(dir)
		if ep, err := os.Executable(); err != nil {
			fmt.Fprint(os.Stderr, "ERROR: ", err)
		} else {
			fmt.Fprint(os.Stderr, ep)
		}
		os.Exit(0)
	}
}
