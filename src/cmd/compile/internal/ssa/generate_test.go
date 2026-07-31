// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"bytes"
	"fmt"
	"internal/testenv"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

const expectedHeader = "// Code generated from _gen/" // this is the common part

// TestGeneratedFilesUpToDate regenerates all the rewrite and rewrite-related
// files defined in _gen into a temporary directory,
// checks that they match what appears in the source tree,
// verifies that they start with the prefix of a generated header,
// and checks that the only source files with that header were actually generated.
func TestGeneratedFilesUpToDate(t *testing.T) {
	testenv.MustHaveGoRun(t)
	wd, err := os.Getwd()
	if err != nil {
		t.Fatalf("Failed to get current working directory: %v", err)
	}
	genDir := filepath.Join(wd, "_gen")
	if _, err := os.Stat(genDir); os.IsNotExist(err) {
		t.Fatalf("_gen directory not found")
	}

	tmpdir := t.TempDir()

	mustRel := func(dir, f string) string {
		rel, err := filepath.Rel(dir, f)
		if err != nil {
			t.Fatalf("could not make %s relative to %s: %v", f, dir, err)
		}
		return rel
	}

	// Accumulate a list of all existing files that look generated.
	// It's an error if this set does not match the set that are
	// generated into tmpdir.
	genFiles := make(map[string]bool)
	genPrefix := []byte(expectedHeader)
	var ssaFiles []string
	roots := []string{
		wd,
		filepath.Join(wd, "../ssacompile"),
		filepath.Join(wd, "../ssarewrite"),
	}
	for _, root := range roots {
		err = filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
			if slices.Contains(roots, path) && os.IsNotExist(err) {
				return nil
			}
			if err != nil {
				return err
			}
			if base := filepath.Base(path); base == "_gen" || base == "testdata" {
				return filepath.SkipDir
			}
			if !d.IsDir() && strings.HasSuffix(path, ".go") {
				ssaFiles = append(ssaFiles, path)
			}
			return nil
		})
		if err != nil {
			t.Fatalf("could not glob for .go files in %s: %v", root, err)
		}
	}
	for _, f := range ssaFiles {
		contents, err := os.ReadFile(f)
		if err != nil {
			t.Fatalf("could not read source file from ssa directory: %v", err)
		}
		// verify that the generated file has the expected header
		// (this should cause other failures later, but if this is
		// the problem, diagnose it here to shorten the treasure hunt.)
		if bytes.HasPrefix(contents, genPrefix) {
			genFiles[mustRel(wd, f)] = true
		}
	}

	goFiles, err := filepath.Glob(filepath.Join(genDir, "*.go"))
	if err != nil {
		t.Fatalf("could not glob for .go files in _gen: %v", err)
	}
	if len(goFiles) == 0 {
		t.Fatal("no .go files found in _gen")
	}

	// Construct the command line for "go run".
	// Explicitly list the files, just to make it
	// clear what is included (if the test is logging).
	args := []string{"run", "-C", genDir}
	for _, f := range goFiles {
		args = append(args, filepath.Base(f))
	}
	args = append(args, "-outdir", filepath.Join(tmpdir, "ssa"))

	logArgs := fmt.Sprintf("%v", args)
	logArgs = logArgs[1 : len(logArgs)-1] // strip '[' and ']'
	t.Logf("%s %v", testenv.GoToolPath(t), logArgs)
	output, err := testenv.Command(t, testenv.GoToolPath(t), args...).CombinedOutput()

	if err != nil {
		t.Fatalf("go run in _gen failed: %v\n%s", err, output)
	}

	// Compare generated files with existing files.
	genRoots := []string{
		filepath.Join(tmpdir, "ssa"),
		filepath.Join(tmpdir, "ssacompile"),
		filepath.Join(tmpdir, "ssarewrite"),
	}
	compare := func(path string, file fs.DirEntry, err error) error {
		if slices.Contains(genRoots, path) && os.IsNotExist(err) {
			// The subpackage hasn't been created yet.
			return nil
		}
		if err != nil {
			return err
		}
		if file.IsDir() {
			return nil
		}
		filename := mustRel(filepath.Join(tmpdir, "ssa"), path)

		// filename must be in the generated set,
		if !genFiles[filename] {
			t.Errorf("%s does not start with the expected header '%s' (if the header was changed the test needs to be updated)",
				filename, expectedHeader)
		}
		genFiles[filename] = false // remove from set

		generatedPath := path
		originalPath := filepath.Join(wd, filename)

		generatedData, err := os.ReadFile(generatedPath)
		if err != nil {
			t.Errorf("could not read generated file %s: %v", path, err)
			return nil
		}

		// there should be a corresponding file in the ssa directory,
		originalData, err := os.ReadFile(originalPath)
		if err != nil {
			if os.IsNotExist(err) {
				t.Errorf("generated file %s was created, but does not exist in the ssa directory. It may need to be added to the repository.", filename)
			} else {
				t.Errorf("could not read original file %s: %v", originalPath, err)
			}
			return nil
		}

		// and the contents of that file should match.
		if !bytes.Equal(originalData, generatedData) {
			t.Errorf("%s is out of date. Please run 'go generate'.", filename)
		}
		return nil
	}
	for _, genRoot := range genRoots {
		if err := filepath.WalkDir(genRoot, compare); err != nil {
			t.Fatalf("could not walk generated directory %s: %v", genRoot, err)
		}
	}

	// the generated set should be empty now.
	for file, notGenerated := range genFiles {
		if notGenerated {
			t.Errorf("%s has the header of a generated file but was not generated", file)
		}
	}
}
