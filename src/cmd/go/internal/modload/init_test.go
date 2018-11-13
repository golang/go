// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modload

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"
)

func TestFindModuleRootIgnoreDir(t *testing.T) {
	// In Plan 9, directories are automatically created in /n.
	// For example, /n/go.mod always exist, but it's a directory.
	// Test that we ignore directories when trying to find go.mod and other config files.

	dir, err := ioutil.TempDir("", "gotest")
	if err != nil {
		t.Fatalf("failed to create temporary directory: %v", err)
	}
	defer os.RemoveAll(dir)
	if err := os.Mkdir(filepath.Join(dir, "go.mod"), os.ModeDir|0755); err != nil {
		t.Fatalf("Mkdir failed: %v", err)
	}
	for _, name := range altConfigs {
		if err := os.MkdirAll(filepath.Join(dir, name), os.ModeDir|0755); err != nil {
			t.Fatalf("MkdirAll failed: %v", err)
		}
	}
	p := filepath.Join(dir, "example")
	if err := os.Mkdir(p, os.ModeDir|0755); err != nil {
		t.Fatalf("Mkdir failed: %v", err)
	}
	if root, _ := FindModuleRoot(p, "", false); root != "" {
		t.Errorf("FindModuleRoot(%q, \"\", false): %q, want empty string", p, root)
	}
	if root, _ := FindModuleRoot(p, "", true); root != "" {
		t.Errorf("FindModuleRoot(%q, \"\", true): %q, want empty string", p, root)
	}
}
