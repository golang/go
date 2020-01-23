// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mod

import (
	"context"
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"

	"golang.org/x/tools/internal/lsp/cache"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/lsp/tests"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/testenv"
)

func TestMain(m *testing.M) {
	testenv.ExitIfSmallMachine()
	os.Exit(m.Run())
}

// TODO(golang/go#36091): This file can be refactored to look like lsp_test.go
// when marker support gets added for go.mod files.
func TestModfileRemainsUnchanged(t *testing.T) {
	ctx := tests.Context(t)
	cache := cache.New(nil)
	session := cache.NewSession()
	options := tests.DefaultOptions()
	options.TempModfile = true
	options.Env = append(os.Environ(), "GOPACKAGESDRIVER=off", "GOROOT=")

	// Make sure to copy the test directory to a temporary directory so we do not
	// modify the test code or add go.sum files when we run the tests.
	folder, err := tests.CopyFolderToTempDir(filepath.Join("testdata", "unchanged"))
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(folder)

	before, err := ioutil.ReadFile(filepath.Join(folder, "go.mod"))
	if err != nil {
		t.Fatal(err)
	}
	_, snapshot, err := session.NewView(ctx, "diagnostics_test", span.FileURI(folder), options)
	if err != nil {
		t.Fatal(err)
	}
	if !hasTempModfile(ctx, snapshot) {
		return
	}
	after, err := ioutil.ReadFile(filepath.Join(folder, "go.mod"))
	if err != nil {
		t.Fatal(err)
	}
	if string(before) != string(after) {
		t.Errorf("the real go.mod file was changed even when tempModfile=true")
	}
}

// TODO(golang/go#36091): This file can be refactored to look like lsp_test.go
// when marker support gets added for go.mod files.
func TestDiagnostics(t *testing.T) {
	ctx := tests.Context(t)
	cache := cache.New(nil)
	session := cache.NewSession()
	options := tests.DefaultOptions()
	options.TempModfile = true
	options.Env = append(os.Environ(), "GOPACKAGESDRIVER=off", "GOROOT=")

	for _, tt := range []struct {
		testdir string
		want    []source.Diagnostic
	}{
		{
			testdir: "indirect",
			want: []source.Diagnostic{
				{
					Message: "golang.org/x/tools should be a direct dependency.",
					Source:  "go mod tidy",
					// TODO(golang/go#36091): When marker support gets added for go.mod files, we
					// can remove these hard coded positions.
					Range:    protocol.Range{Start: getRawPos(4, 62), End: getRawPos(4, 73)},
					Severity: protocol.SeverityWarning,
				},
			},
		},
		{
			testdir: "unused",
			want: []source.Diagnostic{
				{
					Message:  "golang.org/x/tools is not used in this module.",
					Source:   "go mod tidy",
					Range:    protocol.Range{Start: getRawPos(4, 0), End: getRawPos(4, 61)},
					Severity: protocol.SeverityWarning,
				},
			},
		},
		{
			testdir: "invalidrequire",
			want: []source.Diagnostic{
				{
					Message:  "usage: require module/path v1.2.3",
					Source:   "syntax",
					Range:    protocol.Range{Start: getRawPos(4, 0), End: getRawPos(4, 16)},
					Severity: protocol.SeverityError,
				},
			},
		},
		{
			testdir: "invalidgo",
			want: []source.Diagnostic{
				{
					Message:  "usage: go 1.23",
					Source:   "syntax",
					Range:    protocol.Range{Start: getRawPos(2, 0), End: getRawPos(2, 3)},
					Severity: protocol.SeverityError,
				},
			},
		},
		{
			testdir: "unknowndirective",
			want: []source.Diagnostic{
				{
					Message:  "unknown directive: yo",
					Source:   "syntax",
					Range:    protocol.Range{Start: getRawPos(6, 0), End: getRawPos(6, 1)},
					Severity: protocol.SeverityError,
				},
			},
		},
	} {
		t.Run(tt.testdir, func(t *testing.T) {
			// TODO: Once we refactor this to work with go/packages/packagestest. We do not
			// need to copy to a temporary directory.
			folder, err := tests.CopyFolderToTempDir(filepath.Join("testdata", tt.testdir))
			if err != nil {
				t.Fatal(err)
			}
			defer os.RemoveAll(folder)
			_, snapshot, err := session.NewView(ctx, "diagnostics_test", span.FileURI(folder), options)
			if err != nil {
				t.Fatal(err)
			}
			// TODO: Add testing for when the -modfile flag is turned off and we still get diagnostics.
			if !hasTempModfile(ctx, snapshot) {
				return
			}
			reports, err := Diagnostics(ctx, snapshot)
			if err != nil {
				t.Fatal(err)
			}
			if len(reports) != 1 {
				t.Errorf("expected 1 diagnostic, got %d", len(reports))
			}
			for fh, got := range reports {
				if diff := tests.DiffDiagnostics(fh.URI, tt.want, got); diff != "" {
					t.Error(diff)
				}
			}
		})
	}
}

func hasTempModfile(ctx context.Context, snapshot source.Snapshot) bool {
	_, t := snapshot.View().ModFiles()
	return t != ""
}

func getRawPos(line, character int) protocol.Position {
	return protocol.Position{
		Line:      float64(line),
		Character: float64(character),
	}
}
