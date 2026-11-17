// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package eject

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"cmd/go/internal/load"
)

func TestInactiveOwnedDeclarationUsesLoadedVariant(t *testing.T) {
	dir := t.TempDir()
	const source = "//go:build windows\n\npackage owner\n\ntype Hidden struct { value int }\n"
	if err := os.WriteFile(filepath.Join(dir, "type_windows.lgo"), []byte(source), 0666); err != nil {
		t.Fatal(err)
	}
	host := ownedTestMainPackage(dir, "example.com/owner", "owner")
	host.Dir = dir
	host.GoFiles = []string{"type_linux.lgo"}
	host.IgnoredGoFiles = []string{"type_windows.lgo"}
	variant := ownedTestMainPackage(dir, "example.com/owner", "owner")
	variant.Dir = dir
	variant.GoFiles = []string{"type_windows.lgo"}
	variant.IgnoredGoFiles = []string{"type_linux.lgo"}
	plan := ownedTestPlan(dir, host)
	plan.provenance = map[string][]*load.Package{"example.com/owner": {host, variant}}
	condition, testOnly, err := ownedDeclarationCondition(plan, "example.com/owner", "type_windows.lgo")
	if err != nil || condition == nil || condition.String() != "windows" || testOnly {
		t.Fatalf("loaded Windows declaration: condition=%v test=%v error=%v", condition, testOnly, err)
	}
	// A filename alone is never provenance. Without a loader-selected variant,
	// the original ignored-source rejection still applies.
	plan.provenance["example.com/owner"] = []*load.Package{host}
	if _, _, err := ownedDeclarationCondition(plan, "example.com/owner", "type_windows.lgo"); err == nil || !strings.Contains(err.Error(), "not selected") {
		t.Fatalf("unselected declaration accepted: %v", err)
	}
}
