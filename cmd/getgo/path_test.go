// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAppendPath(t *testing.T) {
	tmpd, err := ioutil.TempDir("", "go")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpd)

	if err := os.Setenv("HOME", tmpd); err != nil {
		t.Fatal(err)
	}

	GOPATH := os.Getenv("GOPATH")
	if err := appendToPATH(filepath.Join(GOPATH, "bin")); err != nil {
		t.Fatal(err)
	}

	shellConfig, err := shellConfigFile()
	if err != nil {
		t.Fatal(err)
	}
	b, err := ioutil.ReadFile(shellConfig)
	if err != nil {
		t.Fatal(err)
	}

	expected := "export PATH=" + pathVar + envSeparator + filepath.Join(GOPATH, "bin")
	if strings.TrimSpace(string(b)) != expected {
		t.Fatalf("expected: %q, got %q", expected, strings.TrimSpace(string(b)))
	}

	// Check that appendToPATH is idempotent.
	if err := appendToPATH(filepath.Join(GOPATH, "bin")); err != nil {
		t.Fatal(err)
	}
	b, err = ioutil.ReadFile(shellConfig)
	if err != nil {
		t.Fatal(err)
	}
	if strings.TrimSpace(string(b)) != expected {
		t.Fatalf("expected: %q, got %q", expected, strings.TrimSpace(string(b)))
	}
}
