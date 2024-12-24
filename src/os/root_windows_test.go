// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build windows

package os_test

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

// Verify that Root.Open rejects Windows reserved names.
func TestRootWindowsDeviceNames(t *testing.T) {
	r, err := os.OpenRoot(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	if f, err := r.Open("NUL"); err == nil {
		t.Errorf(`r.Open("NUL") succeeded; want error"`)
		f.Close()
	}
}

// Verify that Root.Open is case-insensitive.
// (The wrong options to NtOpenFile could make operations case-sensitive,
// so this is worth checking.)
func TestRootWindowsCaseInsensitivity(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "file"), nil, 0666); err != nil {
		t.Fatal(err)
	}
	r, err := os.OpenRoot(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	f, err := r.Open("FILE")
	if err != nil {
		t.Fatal(err)
	}
	f.Close()
	if err := r.Remove("FILE"); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(dir, "file")); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("os.Stat(file) after deletion: %v, want ErrNotFound", err)
	}
}
