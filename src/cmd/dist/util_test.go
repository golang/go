// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"errors"
	"io/ioutil"
	"os"
	"path/filepath"
	"regexp"
	"testing"
)

const (
	nonExistent = false
	existent    = true
)

var (
	reDistTool = regexp.MustCompile("go-tool-dist-.*")
)

func TestXworkdir_existingGOTMPDIR(t *testing.T) {
	testXworkdir(t, existent)
}

func TestXworkdir_nonExistentGOTMPDIR(t *testing.T) {
	testXworkdir(t, nonExistent)
}

func testXworkdir(t *testing.T, withExistentGOTMPDIR bool) {
	tempDir, err := ioutil.TempDir("", "")
	if err != nil {
		t.Fatalf("Failed to create a temporary directory: %v", err)
	}
	// clean up GOTMPDIR on exit
	defer os.RemoveAll(tempDir)

	if !withExistentGOTMPDIR {
		// Delete newly created GOTMPDIR to ensure GOTMPDIR doesn't exist when used
		if err := os.RemoveAll(tempDir); err != nil {
			t.Fatalf("Failed to remove newly created temporary directory: %v", err)
		}
	}

	origGOTMPDIR := os.Getenv("GOTMPDIR")
	os.Setenv("GOTMPDIR", tempDir)
	defer os.Setenv("GOTMPDIR", origGOTMPDIR)

	checkGoTmpDir(t, tempDir)
}

func checkGoTmpDir(t *testing.T, tempDir string) {
	var foundDistToolVestiges bool
	xworkdir()
	filepath.Walk(tempDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			t.Fatal(err)
		}
		if info.IsDir() && reDistTool.MatchString(info.Name()) {
			foundDistToolVestiges = true
			return errors.New("existent GOTMPDIR")
		}
		return nil
	})
	if !foundDistToolVestiges {
		t.Fatalf("GOTMPDIR creation failed in `xworkdir` at %s", tempDir)
	}
}
