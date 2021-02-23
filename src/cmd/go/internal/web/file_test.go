// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package web

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"testing"
)

func TestGetFileURL(t *testing.T) {
	const content = "Hello, file!\n"

	f, err := os.CreateTemp("", "web-TestGetFileURL")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(f.Name())

	if _, err := f.WriteString(content); err != nil {
		t.Error(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}

	u, err := urlFromFilePath(f.Name())
	if err != nil {
		t.Fatal(err)
	}

	b, err := GetBytes(u)
	if err != nil {
		t.Fatalf("GetBytes(%v) = _, %v", u, err)
	}
	if string(b) != content {
		t.Fatalf("after writing %q to %s, GetBytes(%v) read %q", content, f.Name(), u, b)
	}
}

func TestGetNonexistentFile(t *testing.T) {
	path, err := filepath.Abs("nonexistent")
	if err != nil {
		t.Fatal(err)
	}

	u, err := urlFromFilePath(path)
	if err != nil {
		t.Fatal(err)
	}

	b, err := GetBytes(u)
	if !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("GetBytes(%v) = %q, %v; want _, fs.ErrNotExist", u, b, err)
	}
}
