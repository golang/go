// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package next

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRemapMultipleEarlierEdits(t *testing.T) {
	shrink := []wireEdit{
		{StartLine: 1, EndLine: 10, Replacement: "x"},
		{StartLine: 25, EndLine: 28, Replacement: "y"},
	}
	if got := remap(shrink, 30); got != 18 {
		t.Fatalf("shrink: got %d, want 18", got)
	}
	grow := []wireEdit{
		{StartLine: 1, EndLine: 2, Replacement: strings.Repeat("z\n", 19) + "z"},
		{StartLine: 15, EndLine: 16, Replacement: "q"},
	}
	if got := remap(grow, 10); got != 28 {
		t.Fatalf("grow: got %d, want 28", got)
	}
	rev := []wireEdit{shrink[1], shrink[0]}
	if got := remap(rev, 30); got != 18 {
		t.Fatalf("reversed: got %d, want 18", got)
	}
}

func TestWalkGoFilesSkipsNestedModule(t *testing.T) {
	dir := t.TempDir()
	write := func(path, body string) {
		t.Helper()
		if err := os.MkdirAll(filepath.Dir(path), 0o777); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(body), 0o666); err != nil {
			t.Fatal(err)
		}
	}
	write(filepath.Join(dir, "go.mod"), "module example.com/root\n\ngo 1.22\n")
	write(filepath.Join(dir, "a.go"), "package root\n")
	write(filepath.Join(dir, "tools", "go.mod"), "module example.com/tools\n\ngo 1.22\n")
	write(filepath.Join(dir, "tools", "b.go"), "package tools\n")

	known := walkGoFiles(dir)
	var rels []string
	for _, rel := range known {
		rels = append(rels, rel)
		if rel == "tools/b.go" || rel == filepath.Join("tools", "b.go") {
			t.Fatalf("descended into nested module, rel=%s known=%#v", rel, known)
		}
	}
	if len(rels) != 1 || rels[0] != "a.go" {
		t.Fatalf("rels=%#v, want [a.go]", rels)
	}
}
