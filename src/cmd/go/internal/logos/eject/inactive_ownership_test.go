// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package eject

import (
	"os"
	"path/filepath"
	"testing"

	"cmd/go/internal/load"
)

func TestInactiveCensusKeepsLoaderBoundaries(t *testing.T) {
	root, outside := t.TempDir(), t.TempDir()
	write := func(path, source string) {
		t.Helper()
		if err := os.MkdirAll(filepath.Dir(path), 0777); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(source), 0666); err != nil {
			t.Fatal(err)
		}
	}
	write(filepath.Join(root, "go.mod"), "module example.com/boundaries\n\ngo 1.27\n\nignore excluded\n")
	write(filepath.Join(root, "host.lgo"), "package boundaries\n")
	write(filepath.Join(root, "unloaded", "value_windows.lgo"), "package unloaded\n")
	write(filepath.Join(root, "excluded", "value_windows.lgo"), "package excluded\n")
	write(filepath.Join(outside, "value_windows.lgo"), "package outside\n")
	if err := os.Symlink(outside, filepath.Join(root, "linked")); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(filepath.Join(outside, "value_windows.lgo"), filepath.Join(root, "linked_windows.lgo")); err != nil {
		t.Fatal(err)
	}
	host := ownedTestMainPackage(root, "example.com/boundaries", "boundaries")
	host.Dir = root
	host.GoFiles = []string{"host.lgo"}
	plan := ownedTestPlan(root, host)
	plan.logos = []*load.Package{host}
	// The active host file is deliberately missing from staging. A failed
	// host emission must not be relabeled as an inactive variant.
	got := collectInactiveLogosFiles(plan, t.TempDir())
	if len(got) != 1 || got[0].pkg.ImportPath != "example.com/boundaries/unloaded" || got[0].name != "value_windows.lgo" {
		t.Fatalf("census crossed ownership or active-load boundary: %v", got)
	}
}
