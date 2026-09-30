// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package simd_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestCopyCheck(t *testing.T) {
	file1 := "simd_emulated.go"
	file2 := filepath.Join("internal", "bridge", "simd_emulated.go")

	b1, err := os.ReadFile(file1)
	if err != nil {
		t.Fatalf("reading %s: %v", file1, err)
	}
	b2, err := os.ReadFile(file2)
	if err != nil {
		t.Fatalf("reading %s: %v", file2, err)
	}

	lines1 := strings.Split(string(b1), "\n")
	lines2 := strings.Split(string(b2), "\n")

	var sawPackage, sawBuild bool
	diffs := 0
	for i := range min(len(lines1), len(lines2)) {
		l1, l2 := lines1[i], lines2[i]
		if l1 == l2 {
			continue
		}
		diffs++
		switch {
		case strings.Contains(l1, "package") && strings.Contains(l2, "package") && !sawPackage:
			sawPackage = true
		case strings.Contains(l1, "//go:build") && strings.Contains(l2, "//go:build") && !sawBuild:
			sawBuild = true
		default:
			t.Errorf("unexpected difference at line %d:\n\t%s: %s\n\t%s: %s", i+1, file1, l1, file2, l2)
			if diffs > 10 {
				t.Fatalf("too many differences, stopping early")
			}
		}
	}

	if len(lines1) != len(lines2) {
		t.Errorf("line count mismatch: %s has %d lines, %s has %d lines", file1, len(lines1), file2, len(lines2))
	}
	if diffs != 2 || !sawPackage || !sawBuild {
		t.Errorf("expected exactly 2 differing lines (one containing \"package\" and one containing \"//go:build\"), got %d diffs (package=%v, build=%v)", diffs, sawPackage, sawBuild)
	}
}
