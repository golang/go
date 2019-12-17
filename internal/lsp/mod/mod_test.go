// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mod_test

import (
	"context"
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"path/filepath"
	"testing"

	"golang.org/x/tools/internal/lsp/cache"
	"golang.org/x/tools/internal/lsp/mod"
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

func TestDiagnostics(t *testing.T) {
	ctx := tests.Context(t)
	cache := cache.New(nil)
	session := cache.NewSession(ctx)
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
					Message:  "golang.org/x/tools should not be an indirect dependency.",
					Source:   "go mod tidy",
					Range:    protocol.Range{Start: getPos(4, 0), End: getPos(4, 61)},
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
					Range:    protocol.Range{Start: getPos(4, 0), End: getPos(4, 61)},
					Severity: protocol.SeverityWarning,
				},
			},
		},
	} {
		t.Run(tt.testdir, func(t *testing.T) {
			// TODO: Once we refactor this to work with go/packages/packagestest. We do not
			// need to copy to a temporary directory.
			// Make sure to copy the test directory to a temporary directory so we do not
			// modify the test code or add go.sum files when we run the tests.
			folder, err := copyToTempDir(filepath.Join("testdata", tt.testdir))
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
			fileID, got, err := mod.Diagnostics(ctx, snapshot)
			if err != nil {
				t.Fatal(err)
			}
			if diff := tests.DiffDiagnostics(fileID.URI, tt.want, got); diff != "" {
				t.Error(diff)
			}
		})
	}
}

func hasTempModfile(ctx context.Context, snapshot source.Snapshot) bool {
	_, t, _ := snapshot.ModFiles(ctx)
	return t != nil
}

func copyToTempDir(folder string) (string, error) {
	if _, err := os.Stat(folder); err != nil {
		return "", err
	}
	dst, err := ioutil.TempDir("", "modfile_test")
	if err != nil {
		return "", err
	}
	fds, err := ioutil.ReadDir(folder)
	if err != nil {
		return "", err
	}
	for _, fd := range fds {
		srcfp := path.Join(folder, fd.Name())
		dstfp := path.Join(dst, fd.Name())
		stat, err := os.Stat(srcfp)
		if err != nil {
			return "", err
		}
		if !stat.Mode().IsRegular() {
			return "", fmt.Errorf("cannot copy non regular file %s", srcfp)
		}
		contents, err := ioutil.ReadFile(srcfp)
		if err != nil {
			return "", err
		}
		ioutil.WriteFile(dstfp, contents, stat.Mode())
	}
	return dst, nil
}

func getPos(line, character int) protocol.Position {
	return protocol.Position{
		Line:      float64(line),
		Character: float64(character),
	}
}
