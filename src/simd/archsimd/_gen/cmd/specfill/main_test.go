// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"simd/archsimd/_gen/gentools"
	"simd/archsimd/_gen/specdoc"
	"simd/archsimd/_gen/specgen"
)

// TestFill verifies that exported declarations in simd and archsimd match
// internal/spec.
func TestFill(t *testing.T) {
	goroot := gentools.DefaultGOROOT()
	specDir := specgen.MustFindSpecDir(goroot)
	specFuncs, err := specgen.Load(specDir, nil)
	if err != nil {
		t.Fatalf("loading spec: %v", err)
	}
	specIdx := specgen.NewIndex(specFuncs)

	dirs := []string{
		filepath.Join(goroot, "src/simd"),
		filepath.Join(goroot, "src/simd/archsimd"),
	}

	for _, dir := range dirs {
		entries, err := os.ReadDir(dir)
		if err != nil {
			t.Fatalf("reading %s: %v", dir, err)
		}
		for _, e := range entries {
			if e.IsDir() || !strings.HasSuffix(e.Name(), ".go") || strings.HasSuffix(e.Name(), "_test.go") {
				continue
			}
			filePath := filepath.Join(dir, e.Name())
			relPath, err := filepath.Rel(goroot, filePath)
			if err != nil {
				relPath = filePath
			}
			src, err := os.ReadFile(filePath)
			if err != nil {
				t.Fatalf("reading %s: %v", relPath, err)
			}

			isGen, err := isGenerated(src)
			if err != nil {
				t.Fatalf("%s: %v", relPath, err)
			}

			filled, err := specdoc.Fill(src, specIdx, specdoc.Options{
				Filename: filepath.ToSlash(relPath),
				// Since we're checking generator output files, we expect there
				// to already be spec-owned doc comments. Don't report these as
				// errors.
				AllowDocRewrite: true,
				// TODO: Docs aren't yet filled by spec in the generators, so
				// leave them alone in the check. Once they are filled by the
				// generators and in hand-written files, drop this flag.
				NoFillDoc: !isGen,
			})
			var rep *specdoc.Report
			if errors.As(err, &rep) {
				t.Errorf("%s diverges from internal/spec:\n%s", relPath, rep)
			} else if err != nil {
				t.Errorf("checking %s: %v", relPath, err)
			} else if !bytes.Equal(src, filled) {
				diff := gentools.Diff(relPath, src, relPath, filled)
				t.Errorf("%s diverges from internal/spec:\n%s", relPath, diff)
			}
		}
	}
}
