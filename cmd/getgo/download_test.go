// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !plan9
// +build !plan9

package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestDownloadGoVersion(t *testing.T) {
	if testing.Short() {
		t.Skipf("Skipping download in short mode")
	}

	tmpd, err := os.MkdirTemp("", "go")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpd)

	if err := downloadGoVersion("go1.8.1", "linux", "amd64", filepath.Join(tmpd, "go")); err != nil {
		t.Fatal(err)
	}

	// Ensure the VERSION file exists.
	vf := filepath.Join(tmpd, "go", "VERSION")
	if _, err := os.Stat(vf); os.IsNotExist(err) {
		t.Fatalf("file %s does not exist and should", vf)
	}
}
