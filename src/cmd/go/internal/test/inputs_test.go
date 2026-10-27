// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package test

import (
	"bytes"
	"cmd/go/internal/load"
	"cmd/go/internal/work"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestComputeTestInputsIDSavesInputs(t *testing.T) {
	root, outside := t.TempDir(), t.TempDir()
	old := time.Now().Add(-time.Hour)
	for _, f := range []string{filepath.Join(root, "a.txt"), filepath.Join(outside, "b.txt")} {
		if err := os.WriteFile(f, nil, 0666); err != nil {
			t.Fatal(err)
		}
		if err := os.Chtimes(f, old, old); err != nil {
			t.Fatal(err)
		}
	}
	a := &work.Action{Package: &load.Package{PackagePublic: load.PackagePublic{Dir: root, Root: root}}}

	in := []string{
		"getenv HOME",
		"chdir " + outside,
		"open b.txt",
		"stat " + filepath.Join(outside, "missing"),
		"chdir " + root,
		"open a.txt",
		"stat " + filepath.Join(root, "missing"),
		"open " + filepath.Join(outside, "b.txt"),
	}
	var log bytes.Buffer
	log.Write(testlogMagic)
	for _, line := range in {
		log.WriteString(line + "\n")
	}
	var inputs bytes.Buffer
	id, err := computeTestInputsID(a, log.Bytes(), &inputs)
	if err != nil {
		t.Fatal(err)
	}

	// Saved with the names the ID was computed from, made absolute.
	want := string(testlogMagic) + in[0] + "\n" + in[1] + "\n" + in[4] + "\n" +
		"open " + filepath.Join(root, "a.txt") + "\n" + in[6] + "\n"
	if got := inputs.String(); got != want {
		t.Errorf("saved test log:\n%s\nwant:\n%s", got, want)
	}
	id2, err := computeTestInputsID(a, inputs.Bytes(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if id2 != id {
		t.Errorf("inputs ID of the saved log = %x, want %x", id2, id)
	}
}
