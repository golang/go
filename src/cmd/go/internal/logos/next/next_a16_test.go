// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package next

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"testing"
)

func TestA16WalkSkipsOutsideSymlink(t *testing.T) {
	dir := t.TempDir()
	outside := t.TempDir()
	write := func(path, body string) {
		t.Helper()
		if err := os.MkdirAll(filepath.Dir(path), 0o777); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(body), 0o666); err != nil {
			t.Fatal(err)
		}
	}
	write(filepath.Join(dir, "go.mod"), "module example.com/s1\n\ngo 1.22\n")
	write(filepath.Join(dir, "a.go"), "package s1\n")
	write(filepath.Join(outside, "real.go"), "package s1\n")
	target := filepath.Join(outside, "real.go")
	link := filepath.Join(dir, "link.go")
	if err := os.Symlink(target, link); err != nil {
		t.Skip("symlink unavailable")
	}
	known := walkGoFiles(dir)
	for _, rel := range known {
		if rel == "link.go" {
			t.Fatalf("outside symlink collected")
		}
	}
	if len(known) != 1 {
		t.Fatalf("got %d files", len(known))
	}
}

func TestA16ApplySumOk(t *testing.T) {
	data := []byte("package p\n")
	h := sha256.Sum256(data)
	sum := hex.EncodeToString(h[:])
	if !applySumOk(data, sum) {
		t.Fatalf("match rejected")
	}
	bad := []byte("package q\n")
	if applySumOk(bad, sum) {
		t.Fatalf("mismatch accepted")
	}
}

func TestA16HasLgoFiles(t *testing.T) {
	dir := t.TempDir()
	if hasLgoFiles(dir) {
		t.Fatalf("empty reported lgo")
	}
	if err := os.WriteFile(filepath.Join(dir, "a.lgo"), []byte("package p\n"), 0o666); err != nil {
		t.Fatal(err)
	}
	if !hasLgoFiles(dir) {
		t.Fatalf("missing lgo")
	}
}
