// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

// TestPlanConsistent checks SPEC-TRANSITION.md against itself. The document
// states its dependencies three times -- Part 3's table, Part 4's
// "Needs:"/"Prefers:" lines, and Part 4's "Blocks:" lines, which are the same
// edges reversed -- and this is what keeps the three from drifting. It also
// checks that the table is topologically sorted (treating Prefers as edges),
// that unrelated tasks are ordered by track, and that every task is in exactly
// one track.
//
// This runs on the checked-in document rather than on a fixture, because the
// checked-in document is the thing that has to be right.
func TestPlanConsistent(t *testing.T) {
	md, _ := paths("")
	for _, p := range load(md).problems() {
		t.Errorf("%s: %s", filepath.Base(md), p)
	}
}

// TestGraphUpToDate checks that SPEC-TASKS.dot still matches the document it
// was generated from, so that editing the plan without regenerating is caught
// here rather than noticed later in a stale picture.
func TestGraphUpToDate(t *testing.T) {
	md, dotPath := paths("")
	var want bytes.Buffer
	load(md).dot(&want)

	got, err := os.ReadFile(dotPath)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want.Bytes()) {
		t.Errorf("%s is out of date; regenerate with:\n\tgo run ./cmd/spectasks -w",
			filepath.Base(dotPath))
	}
}
