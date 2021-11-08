// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build cgo

package pe

import (
	"os/exec"
	"runtime"
	"testing"
)

func testCgoDWARF(t *testing.T, linktype int) {
	if _, err := exec.LookPath("gcc"); err != nil {
		t.Skip("skipping test: gcc is missing")
	}
	testDWARF(t, linktype)
}

func TestDefaultLinkerDWARF(t *testing.T) {
	testCgoDWARF(t, linkCgoDefault)
}

func TestInternalLinkerDWARF(t *testing.T) {
	if runtime.GOARCH == "arm64" {
		t.Skip("internal linker disabled on windows/arm64")
	}
	testCgoDWARF(t, linkCgoInternal)
}

func TestExternalLinkerDWARF(t *testing.T) {
	testCgoDWARF(t, linkCgoExternal)
}
