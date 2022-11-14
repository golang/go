// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package coverage

import (
	"internal/goexperiment"
	"os"
	"path/filepath"
	"strings"
	"testing"
	_ "unsafe"
)

//go:linkname testing_testGoCoverDir testing.testGoCoverDir
func testing_testGoCoverDir() string

// TestTestSupport does a basic verification of the functionality in
// runtime/coverage.processCoverTestDir (doing this here as opposed to
// relying on other test paths will provide a better signal when
// running "go test -cover" for this package).
func TestTestSupport(t *testing.T) {
	if !goexperiment.CoverageRedesign {
		return
	}
	if testing.CoverMode() == "" {
		return
	}
	t.Logf("testing.testGoCoverDir() returns %s mode=%s\n",
		testing_testGoCoverDir(), testing.CoverMode())

	textfile := filepath.Join(t.TempDir(), "file.txt")
	var sb strings.Builder
	err := processCoverTestDirInternal(testing_testGoCoverDir(), textfile,
		testing.CoverMode(), "", &sb)
	if err != nil {
		t.Fatalf("bad: %v", err)
	}

	// Check for existence of text file.
	if inf, err := os.Open(textfile); err != nil {
		t.Fatalf("problems opening text file %s: %v", textfile, err)
	} else {
		inf.Close()
	}

	// Check for percent output with expected tokens.
	strout := sb.String()
	want1 := "runtime/coverage"
	want2 := "of statements"
	if !strings.Contains(strout, want1) ||
		!strings.Contains(strout, want2) {
		t.Logf("output from run: %s\n", strout)
		t.Fatalf("percent output missing key tokens: %q and %q",
			want1, want2)
	}
}
