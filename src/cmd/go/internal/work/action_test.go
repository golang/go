// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package work

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
	reBuild = regexp.MustCompile("go-build")
)

func TestBuilder_existingGOTMPDIR(t *testing.T) {
	testBuilder(t, existent)
}

func TestBuilder_nonExistentGOTMPDIR(t *testing.T) {
	testBuilder(t, nonExistent)
}

func testBuilder(t *testing.T, withExistentGOTMPDIR bool) {
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
	var foundGOBUILDDIR bool
	b := Builder{}
	b.Init()
	filepath.Walk(tempDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			t.Fatal(err)
		}
		if info.IsDir() && reBuild.MatchString(info.Name()) {
			foundGOBUILDDIR = true
			return errors.New("existent GOTMPDIR")
		}
		return nil
	})
	if !foundGOBUILDDIR {
		t.Fatalf("Error: GOTMPDIR creation failed in `Builder.Init` at %s", tempDir)
	}
}
