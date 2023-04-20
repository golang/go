// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mod

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"

	"golang.org/x/tools/gopls/internal/lsp/cache"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/gopls/internal/lsp/tests"
	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/internal/testenv"
)

func TestMain(m *testing.M) {
	testenv.ExitIfSmallMachine()
	os.Exit(m.Run())
}

func TestModfileRemainsUnchanged(t *testing.T) {
	testenv.NeedsExec(t)

	ctx := tests.Context(t)
	session := cache.NewSession(ctx, cache.New(nil), nil)
	options := source.DefaultOptions().Clone()
	tests.DefaultOptions(options)
	options.TempModfile = true
	options.Env = map[string]string{"GOPACKAGESDRIVER": "off", "GOROOT": ""}

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
	_, _, release, err := session.NewView(ctx, "diagnostics_test", span.URIFromPath(folder), options)
	if err != nil {
		t.Fatal(err)
	}
	release()
	after, err := ioutil.ReadFile(filepath.Join(folder, "go.mod"))
	if err != nil {
		t.Fatal(err)
	}
	if string(before) != string(after) {
		t.Errorf("the real go.mod file was changed even when tempModfile=true")
	}
}
