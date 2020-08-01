// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !android,!js,!ppc64le

// Note: we don't run on Android or ppc64 because if there is any non-race test
// file in this package, the OS tries to link the .syso file into the
// test (even when we're not in race mode), which fails. I'm not sure
// why, but easiest to just punt - as long as a single builder runs
// this test, we're good.

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
