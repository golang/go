// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"internal/testenv"
	"io/fs"
	"os"
	"path/filepath"
	"testing"

	"golang.org/x/build/relnote"
)

var flagCheck = flag.Bool("check", false, "run API release note checks")

// Check that each file in api/next has corresponding release note files in doc/next.
func TestCheckAPIFragments(t *testing.T) {
	if !*flagCheck {
		t.Skip("-check not specified")
	}
	root := testenv.GOROOT(t)
	rootFS := os.DirFS(root)
	files, err := fs.Glob(rootFS, "api/next/*.txt")
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("checking release notes for %d files in api/next", len(files))
	docFS := os.DirFS(filepath.Join(root, "doc", "next"))
	// Check that each api/next file has a corresponding release note fragment.
	for _, apiFile := range files {
		if err := relnote.CheckAPIFile(rootFS, apiFile, docFS, "doc/next"); err != nil {
			t.Errorf("%s: %v", apiFile, err)
		}
	}
}
