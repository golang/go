// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements
// TestBuilderWithExistingGoTmpDir; a test to check successful Builder.Init() call if GOTMPDIR exists
// TestBuilderWithoutExistingGoTmpDir; a test to check successful Builder.Init() call if GOTMPDIR does not exists
//

package work

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

func TestBuilderWithExistingGoTmpDir(t *testing.T) {
	tempDir, err := ioutil.TempDir("", "")
	if err != nil {
		fmt.Println(err)
	}
	defer os.RemoveAll(tempDir)

	os.Setenv("GOTMPDIR", tempDir)
	defer os.Setenv("GOTMPDIR", EnvGoTmpDir)

	checkGoTmpDir(t, tempDir)
}

func TestBuilderWithoutExistingGoTmpDir(t *testing.T) {
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
	r, _ := regexp.Compile("go-build")

	_ := Builder{}

	var goTmpDirCreated bool
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
		t.Fatalf("Error: GOTMPDIR creation failed in `Builder.Init` at %s", tempDir)
	}
}
