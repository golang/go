// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"os"
	"testing"

	"golang.org/x/tools/internal/lsp/fake"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
)

// osFileSource is a fileSource that just reads from the operating system.
type osFileSource struct{}

func (s osFileSource) GetFile(ctx context.Context, uri span.URI) (source.FileHandle, error) {
	fi, statErr := os.Stat(uri.Filename())
	if statErr != nil {
		return &fileHandle{
			err: statErr,
			uri: uri,
		}, nil
	}
	fh, err := readFile(ctx, uri, fi.ModTime())
	if err != nil {
		return nil, err
	}
	return fh, nil
}

func TestWorkspaceModule(t *testing.T) {
	tests := []struct {
		desc           string
		initial        string // txtar-encoded
		legacyMode     bool
		initialSource  workspaceSource
		initialModules []string
		initialDirs    []string
		updates        map[string]string
		finalSource    workspaceSource
		finalModules   []string
		finalDirs      []string
	}{
		{
			desc: "legacy mode",
			initial: `
-- go.mod --
module mod.com
-- a/go.mod --
module moda.com`,
			legacyMode:     true,
			initialModules: []string{"./go.mod"},
			initialSource:  legacyWorkspace,
			initialDirs:    []string{"."},
		},
		{
			desc: "nested module",
			initial: `
-- go.mod --
module mod.com
-- a/go.mod --
module moda.com`,
			initialModules: []string{"./go.mod", "a/go.mod"},
			initialSource:  fileSystemWorkspace,
			initialDirs:    []string{".", "a"},
		},
		{
			desc: "removing module",
			initial: `
-- a/go.mod --
module moda.com
-- b/go.mod --
module modb.com`,
			initialModules: []string{"a/go.mod", "b/go.mod"},
			initialSource:  fileSystemWorkspace,
			initialDirs:    []string{".", "a", "b"},
			updates: map[string]string{
				"gopls.mod": `module gopls-workspace

require moda.com v0.0.0-goplsworkspace
replace moda.com => $SANDBOX_WORKDIR/a`,
			},
			finalModules: []string{"a/go.mod"},
			finalSource:  goplsModWorkspace,
			finalDirs:    []string{".", "a"},
		},
		{
			desc: "adding module",
			initial: `
-- gopls.mod --
require moda.com v0.0.0-goplsworkspace
replace moda.com => $SANDBOX_WORKDIR/a
-- a/go.mod --
module moda.com
-- b/go.mod --
module modb.com`,
			initialModules: []string{"a/go.mod"},
			initialSource:  goplsModWorkspace,
			initialDirs:    []string{".", "a"},
			updates: map[string]string{
				"gopls.mod": `module gopls-workspace

require moda.com v0.0.0-goplsworkspace
require modb.com v0.0.0-goplsworkspace

replace moda.com => $SANDBOX_WORKDIR/a
replace modb.com => $SANDBOX_WORKDIR/b`,
			},
			finalModules: []string{"a/go.mod", "b/go.mod"},
			finalSource:  goplsModWorkspace,
			finalDirs:    []string{".", "a", "b"},
		},
		{
			desc: "deleting gopls.mod",
			initial: `
-- gopls.mod --
module gopls-workspace

require moda.com v0.0.0-goplsworkspace
replace moda.com => $SANDBOX_WORKDIR/a
-- a/go.mod --
module moda.com
-- b/go.mod --
module modb.com`,
			initialModules: []string{"a/go.mod"},
			initialSource:  goplsModWorkspace,
			initialDirs:    []string{".", "a"},
			updates: map[string]string{
				"gopls.mod": "",
			},
			finalModules: []string{"a/go.mod", "b/go.mod"},
			finalSource:  fileSystemWorkspace,
			finalDirs:    []string{".", "a", "b"},
		},
		{
			desc: "broken module parsing",
			initial: `
-- a/go.mod --
module moda.com

require gopls.test v0.0.0-goplsworkspace
replace gopls.test => ../../gopls.test // (this path shouldn't matter)
-- b/go.mod --
module modb.com`,
			initialModules: []string{"a/go.mod", "b/go.mod"},
			initialSource:  fileSystemWorkspace,
			initialDirs:    []string{".", "a", "b", "../gopls.test"},
			updates: map[string]string{
				"a/go.mod": `modul moda.com

require gopls.test v0.0.0-goplsworkspace
replace gopls.test => ../../gopls.test2`,
			},
			finalModules: []string{"a/go.mod", "b/go.mod"},
			finalSource:  fileSystemWorkspace,
			// finalDirs should be unchanged: we should preserve dirs in the presence
			// of a broken modfile.
			finalDirs: []string{".", "a", "b", "../gopls.test"},
		},
	}

	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			ctx := context.Background()
			dir, err := fake.Tempdir(test.initial)
			if err != nil {
				t.Fatal(err)
			}
			defer os.RemoveAll(dir)
			root := span.URIFromPath(dir)

			fs := osFileSource{}
			wm, err := newWorkspace(ctx, root, fs, !test.legacyMode)
			if err != nil {
				t.Fatal(err)
			}
			rel := fake.RelativeTo(dir)
			checkWorkspaceModule(t, rel, wm, test.initialSource, test.initialModules)
			gotDirs := wm.dirs(ctx, fs)
			checkWorkspaceDirs(t, rel, gotDirs, test.initialDirs)
			if test.updates != nil {
				changes := make(map[span.URI]*fileChange)
				for k, v := range test.updates {
					if v == "" {
						// for convenience, use this to signal a deletion. TODO: more doc
						err := os.Remove(rel.AbsPath(k))
						if err != nil {
							t.Fatal(err)
						}
					} else {
						fake.WriteFileData(k, []byte(v), rel)
					}
					uri := span.URIFromPath(rel.AbsPath(k))
					fh, err := fs.GetFile(ctx, uri)
					if err != nil {
						t.Fatal(err)
					}
					content, err := fh.Read()
					changes[uri] = &fileChange{
						content:    content,
						exists:     err == nil,
						fileHandle: &closedFile{fh},
					}
				}
				wm, _ := wm.invalidate(ctx, changes)
				checkWorkspaceModule(t, rel, wm, test.finalSource, test.finalModules)
				gotDirs := wm.dirs(ctx, fs)
				checkWorkspaceDirs(t, rel, gotDirs, test.finalDirs)
			}
		})
	}
}

func checkWorkspaceModule(t *testing.T, rel fake.RelativeTo, got *workspace, wantSource workspaceSource, want []string) {
	t.Helper()
	if got.moduleSource != wantSource {
		t.Errorf("module source = %v, want %v", got.moduleSource, wantSource)
	}
	modules := make(map[span.URI]struct{})
	for k := range got.activeModFiles() {
		modules[k] = struct{}{}
	}
	for _, modPath := range want {
		path := rel.AbsPath(modPath)
		uri := span.URIFromPath(path)
		if _, ok := modules[uri]; !ok {
			t.Errorf("missing module %q", uri)
		}
		delete(modules, uri)
	}
	for remaining := range modules {
		t.Errorf("unexpected module %q", remaining)
	}
}

func checkWorkspaceDirs(t *testing.T, rel fake.RelativeTo, got []span.URI, want []string) {
	t.Helper()
	gotM := make(map[span.URI]bool)
	for _, dir := range got {
		gotM[dir] = true
	}
	for _, dir := range want {
		path := rel.AbsPath(dir)
		uri := span.URIFromPath(path)
		if !gotM[uri] {
			t.Errorf("missing dir %q", uri)
		}
		delete(gotM, uri)
	}
	for remaining := range gotM {
		t.Errorf("unexpected dir %q", remaining)
	}
}
