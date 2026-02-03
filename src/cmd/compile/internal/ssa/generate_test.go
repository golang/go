// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"bytes"
	"fmt"
	"internal/testenv"
	"os"
	"path/filepath"
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

	// Accumulate a list of all existing files that look generated.
	// It's an error if this set does not match the set that are
	// generated into tmpdir.
	genFiles := make(map[string]bool)
	genPrefix := []byte(expectedHeader)
	ssaFiles, err := filepath.Glob(filepath.Join(wd, "*.go"))
	if err != nil {
		t.Fatalf("could not glob for .go files in ssa directory: %v", err)
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
			genFiles[filepath.Base(f)] = true
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
	args = append(args, "-outdir", tmpdir)

	logArgs := fmt.Sprintf("%v", args)
	logArgs = logArgs[1 : len(logArgs)-2] // strip '[' and ']'
	t.Logf("%s %v", testenv.GoToolPath(t), logArgs)
	output, err := testenv.Command(t, testenv.GoToolPath(t), args...).CombinedOutput()

	if err != nil {
		t.Fatalf("go run in _gen failed: %v\n%s", err, output)
	}

	// Compare generated files with existing files in the parent directory.
	files, err := os.ReadDir(tmpdir)
	if err != nil {
		t.Fatalf("could not read tmpdir %s: %v", tmpdir, err)
	}

	for _, file := range files {
		if file.IsDir() {
			continue
		}
		filename := file.Name()

		// filename must be in the generated set,
		if !genFiles[filename] {
			t.Errorf("%s does not start with the expected header '%s' (if the header was changed the test needs to be updated)",
				filename, expectedHeader)
		}
		genFiles[filename] = false // remove from set

		generatedPath := filepath.Join(tmpdir, filename)
		originalPath := filepath.Join(wd, filename)

		generatedData, err := os.ReadFile(generatedPath)
		if err != nil {
			t.Errorf("could not read generated file %s: %v", generatedPath, err)
			continue
		}

		// there should be a corresponding file in the ssa directory,
		originalData, err := os.ReadFile(originalPath)
		if err != nil {
			if os.IsNotExist(err) {
				t.Errorf("generated file %s was created, but does not exist in the ssa directory. It may need to be added to the repository.", filename)
			} else {
				t.Errorf("could not read original file %s: %v", originalPath, err)
			}
			continue
		}

		// and the contents of that file should match.
		if !bytes.Equal(originalData, generatedData) {
			t.Errorf("%s is out of date. Please run 'go generate'.", filename)
		}
	}

	// the generated set should be empty now.
	for file, notGenerated := range genFiles {
		if notGenerated {
			t.Errorf("%s has the header of a generated file but was not generated", file)
		}
	}
}
