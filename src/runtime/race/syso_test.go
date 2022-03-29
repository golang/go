// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build race

package race

import (
	"bytes"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
)

func TestIssue37485(t *testing.T) {
	files, err := filepath.Glob("./*.syso")
	if err != nil {
		t.Fatalf("can't find syso files: %s", err)
	}
	for _, f := range files {
		cmd := exec.Command(filepath.Join(runtime.GOROOT(), "bin", "go"), "tool", "nm", f)
		res, err := cmd.CombinedOutput()
		if err != nil {
			t.Errorf("nm of %s failed: %s", f, err)
			continue
		}
		if bytes.Contains(res, []byte("getauxval")) {
			t.Errorf("%s contains getauxval", f)
		}
	}
}
