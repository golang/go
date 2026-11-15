// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cpu_test

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
)

func TestLogosCacheLinePadSizeCrossArch(t *testing.T) {
	gobin := filepath.Join(runtime.GOROOT(), "bin", "go")
	if _, err := os.Stat(gobin); err != nil {
		t.Skipf("go tool not found: %v", err)
	}
	for _, goarch := range []string{"amd64", "arm64"} {
		out := filepath.Join(t.TempDir(), "padprobe")
		cmd := exec.Command(gobin, "build", "-o", out, "./testdata/probe")
		cmd.Dir = filepath.Join(runtime.GOROOT(), "src", "internal", "cpu")
		cmd.Env = append(os.Environ(),
			"GOARCH="+goarch,
			"GOOS=linux",
			"CGO_ENABLED=0",
			"GOPROXY=off",
			"GOTOOLCHAIN=local",
		)
		if text, err := cmd.CombinedOutput(); err != nil {
			t.Errorf("GOARCH=%s probe build: %v\n%s", goarch, err, text)
		}
	}
}
