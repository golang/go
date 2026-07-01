// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package simd_test

import (
	"internal/testenv"
	"os"
	"runtime"
	"strings"
	"testing"
)

func common(t *testing.T, dir, what, failWith string, moreEnv ...string) {
	t.Helper()
	t.Logf("subprocess test in testdata")
	testenv.MustHaveGoRun(t)
	args := []string{"test", "-C", dir}
	if testing.Verbose() {
		args = append(args, "-v")
	}
	args = append(args, what)
	cmd := testenv.Command(t, testenv.GoToolPath(t), args...)

	goexp := os.Getenv("GOEXPERIMENT")
	if !strings.Contains(","+goexp+",", ",simd,") {
		if goexp != "" {
			goexp += ","
		}
		goexp += "simd"
	}
	cmd.Env = append(cmd.Environ(), "GOEXPERIMENT="+goexp)
	cmd.Env = append(cmd.Env, moreEnv...)

	if failWith == "" {
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		if err := cmd.Run(); err != nil {
			t.Error(err)
		}
	} else {
		combined, err := cmd.CombinedOutput()
		combinedString := string(combined)
		sawFailure := strings.Contains(combinedString, failWith)
		if err == nil && !sawFailure {
			t.Errorf("Saw no error and did not see expected failure string '%s' in '%s'", failWith, combinedString)
		} else if err == nil && sawFailure {
			t.Errorf("Saw no error but did see expected failure string '%s'", failWith)
		} else if err != nil && !sawFailure {
			t.Errorf("Saw error %v but did see expected failure string '%s' in '%s'", err, failWith, combinedString)
		} else /* err != nil && sawFailure */ {
			t.Logf("Saw error %v and expected failure string '%s'", err, failWith)
		}
	}
}
func TestIFace(t *testing.T) {
	common(t, "testdata", "iface_test.go", "")
}

func TestSizeof(t *testing.T) {
	common(t, "testdata", "sizeof_test.go", "")
}

func TestCompileOk(t *testing.T) {
	common(t, "testdata", "compiles_test.go", "")
}

func TestCompileError(t *testing.T) {
	common(t, "testdata", "errors_test.go",
		"array length unsafe.Sizeof(v_from_simd) (value of type uintptr) must be constant")
}

func TestToString(t *testing.T) {
	common(t, "testdata", "tostring_test.go", "")
	if runtime.GOARCH == "amd64" || runtime.GOARCH == "arm64" {
		common(t, "testdata", "tostring_test.go", "", "GODEBUG=simd=0")
	}
}
