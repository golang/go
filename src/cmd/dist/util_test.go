// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements
// TestXworkdirWithExistingGoTmpDir; a test to check successful xworkdir() call if GOTMPDIR exists
// TestXworkdirWithoutExistingGoTmpDir; a test to check successful xworkdir() call if GOTMPDIR does not exists
//

package main

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"regexp"
	"testing"
)

var (
	EnvGoTmpDir = os.Getenv("GOTMPDIR")
)

func TestXworkdirWithExistingGoTmpDir(t *testing.T) {
	tempDir, err := ioutil.TempDir("", "")
	if err != nil {
		fmt.Println(err)
	}
	defer os.RemoveAll(tempDir)

	os.Setenv("GOTMPDIR", tempDir)
	defer os.Setenv("GOTMPDIR", EnvGoTmpDir)

	checkGoTmpDir(t, tempDir)
}

func TestXworkdirWithoutExistingGoTmpDir(t *testing.T) {
	tempDir, err := ioutil.TempDir("", "")
	if err != nil {
		fmt.Println(err)
	}
	os.RemoveAll(tempDir)

	os.Setenv("GOTMPDIR", tempDir)
	defer os.Setenv("GOTMPDIR", EnvGoTmpDir)

	checkGoTmpDir(t, tempDir)
}

func checkGoTmpDir(t *testing.T, tempDir string) {
	r, _ := regexp.Compile("go-tool-dist-.*")

	var goTmpDirCreated bool
	xworkdir()
	filepath.Walk(tempDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			t.Fatal(err)
		}
		if info.IsDir() && r.MatchString(info.Name()) {
			goTmpDirCreated = true
		}
		return nil
	})
	if !goTmpDirCreated {
		t.Fatalf("Error: GOTMPDIR creation failed in `xworkdir` at %s", tempDir)
	}
}
