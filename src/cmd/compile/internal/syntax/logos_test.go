// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import (
	"bytes"
	"io/fs"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// dumpAddr matches the raw pointer values that Fdump prints for slices
// of node pointers; they differ between any two parses.
var dumpAddr = regexp.MustCompile(`0x[0-9a-f]+`)

// TestLogosModeParsesGo checks that LogosMode parses plain Go exactly
// like the default mode: every .go file under testdata yields identical
// errors and an identical Fdump under CheckBranches|LogosMode and
// CheckBranches. Files that fail in plain mode must fail identically,
// never be skipped silently.
func TestLogosModeParsesGo(t *testing.T) {
	var files []string
	err := filepath.WalkDir("testdata", func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() && strings.HasSuffix(d.Name(), ".go") {
			files = append(files, path)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(files) == 0 {
		t.Fatal("no .go files in testdata")
	}
	for _, filename := range files {
		var plainErrs, logosErrs []string
		plainFile, _ := ParseFile(filename, func(err error) { plainErrs = append(plainErrs, err.Error()) }, nil, CheckBranches)
		logosFile, _ := ParseFile(filename, func(err error) { logosErrs = append(logosErrs, err.Error()) }, nil, CheckBranches|LogosMode)
		if len(plainErrs) != len(logosErrs) {
			t.Errorf("%s: error count differs: plain=%d logos=%d\nplain: %q\nlogos: %q",
				filename, len(plainErrs), len(logosErrs), plainErrs, logosErrs)
			continue
		}
		for i := range plainErrs {
			if plainErrs[i] != logosErrs[i] {
				t.Errorf("%s: error %d differs:\nplain: %s\nlogos: %s",
					filename, i, plainErrs[i], logosErrs[i])
			}
		}
		if plainFile == nil || logosFile == nil {
			if (plainFile == nil) != (logosFile == nil) {
				t.Errorf("%s: nil AST differs: plain nil=%v logos nil=%v",
					filename, plainFile == nil, logosFile == nil)
			}
			continue
		}
		var plainDump, logosDump bytes.Buffer
		if err := Fdump(&plainDump, plainFile); err != nil {
			t.Errorf("%s: plain Fdump: %v", filename, err)
			continue
		}
		if err := Fdump(&logosDump, logosFile); err != nil {
			t.Errorf("%s: logos Fdump: %v", filename, err)
			continue
		}
		if dumpAddr.ReplaceAllString(plainDump.String(), "0x") != dumpAddr.ReplaceAllString(logosDump.String(), "0x") {
			t.Errorf("%s: Fdump differs under LogosMode", filename)
		}
	}
}
