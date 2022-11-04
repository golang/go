// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"errors"
	"os"
	"strings"
	"testing"

	"golang.org/x/mod/modfile"
	"golang.org/x/tools/gopls/internal/lsp/fake"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/gopls/internal/span"
)

// osFileSource is a fileSource that just reads from the operating system.
type osFileSource struct {
	overlays map[span.URI]fakeOverlay
}

type fakeOverlay struct {
	source.VersionedFileHandle
	uri     span.URI
	content string
	err     error
	saved   bool
}

func (o fakeOverlay) Saved() bool { return o.saved }

func (o fakeOverlay) Read() ([]byte, error) {
	if o.err != nil {
		return nil, o.err
	}
	return []byte(o.content), nil
}

func (o fakeOverlay) URI() span.URI {
	return o.uri
}

// change updates the file source with the given file content. For convenience,
// empty content signals a deletion. If saved is true, these changes are
// persisted to disk.
func (s *osFileSource) change(ctx context.Context, uri span.URI, content string, saved bool) (*fileChange, error) {
	if content == "" {
		delete(s.overlays, uri)
		if saved {
			if err := os.Remove(uri.Filename()); err != nil {
				return nil, err
			}
		}
		fh, err := s.GetFile(ctx, uri)
		if err != nil {
			return nil, err
		}
		data, err := fh.Read()
		return &fileChange{exists: err == nil, content: data, fileHandle: &closedFile{fh}}, nil
	}
	if s.overlays == nil {
		s.overlays = map[span.URI]fakeOverlay{}
	}
	s.overlays[uri] = fakeOverlay{uri: uri, content: content, saved: saved}
	return &fileChange{
		exists:     content != "",
		content:    []byte(content),
		fileHandle: s.overlays[uri],
	}, nil
}

func (s *osFileSource) GetFile(ctx context.Context, uri span.URI) (source.FileHandle, error) {
	if overlay, ok := s.overlays[uri]; ok {
		return overlay, nil
	}
	fi, statErr := os.Stat(uri.Filename())
	if statErr != nil {
		return &fileHandle{
			err: statErr,
			uri: uri,
		}, nil
	}
	fh, err := readFile(ctx, uri, fi)
	if err != nil {
		return nil, err
	}
	return fh, nil
}

type wsState struct {
	source  workspaceSource
	modules []string
	dirs    []string
	sum     string
}

type wsChange struct {
	content string
	saved   bool
}

func TestWorkspaceModule(t *testing.T) {
	tests := []struct {
		desc         string
		initial      string // txtar-encoded
		legacyMode   bool
		initialState wsState
		updates      map[string]wsChange
		wantChanged  bool
		wantReload   bool
		finalState   wsState
	}{
		{
			desc: "legacy mode",
			initial: `
-- go.mod --
module mod.com
-- go.sum --
golang.org/x/mod v0.3.0 h1:deadbeef
-- a/go.mod --
module moda.com`,
			legacyMode: true,
			initialState: wsState{
				modules: []string{"./go.mod"},
				source:  legacyWorkspace,
				dirs:    []string{"."},
				sum:     "golang.org/x/mod v0.3.0 h1:deadbeef\n",
			},
		},
		{
			desc: "nested module",
			initial: `
-- go.mod --
module mod.com
-- a/go.mod --
module moda.com`,
			initialState: wsState{
				modules: []string{"./go.mod", "a/go.mod"},
				source:  fileSystemWorkspace,
				dirs:    []string{".", "a"},
			},
		},
		{
			desc: "removing module",
			initial: `
-- a/go.mod --
module moda.com
-- a/go.sum --
golang.org/x/mod v0.3.0 h1:deadbeef
-- b/go.mod --
module modb.com
-- b/go.sum --
golang.org/x/mod v0.3.0 h1:beefdead`,
			initialState: wsState{
				modules: []string{"a/go.mod", "b/go.mod"},
				source:  fileSystemWorkspace,
				dirs:    []string{".", "a", "b"},
				sum:     "golang.org/x/mod v0.3.0 h1:beefdead\ngolang.org/x/mod v0.3.0 h1:deadbeef\n",
			},
			updates: map[string]wsChange{
				"gopls.mod": {`module gopls-workspace

require moda.com v0.0.0-goplsworkspace
replace moda.com => $SANDBOX_WORKDIR/a`, true},
			},
			wantChanged: true,
			wantReload:  true,
			finalState: wsState{
				modules: []string{"a/go.mod"},
				source:  goplsModWorkspace,
				dirs:    []string{".", "a"},
				sum:     "golang.org/x/mod v0.3.0 h1:deadbeef\n",
			},
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
			initialState: wsState{
				modules: []string{"a/go.mod"},
				source:  goplsModWorkspace,
				dirs:    []string{".", "a"},
			},
			updates: map[string]wsChange{
				"gopls.mod": {`module gopls-workspace

require moda.com v0.0.0-goplsworkspace
require modb.com v0.0.0-goplsworkspace

replace moda.com => $SANDBOX_WORKDIR/a
replace modb.com => $SANDBOX_WORKDIR/b`, true},
			},
			wantChanged: true,
			wantReload:  true,
			finalState: wsState{
				modules: []string{"a/go.mod", "b/go.mod"},
				source:  goplsModWorkspace,
				dirs:    []string{".", "a", "b"},
			},
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
			initialState: wsState{
				modules: []string{"a/go.mod"},
				source:  goplsModWorkspace,
				dirs:    []string{".", "a"},
			},
			updates: map[string]wsChange{
				"gopls.mod": {"", true},
			},
			wantChanged: true,
			wantReload:  true,
			finalState: wsState{
				modules: []string{"a/go.mod", "b/go.mod"},
				source:  fileSystemWorkspace,
				dirs:    []string{".", "a", "b"},
			},
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
			initialState: wsState{
				modules: []string{"a/go.mod", "b/go.mod"},
				source:  fileSystemWorkspace,
				dirs:    []string{".", "a", "b", "../gopls.test"},
			},
			updates: map[string]wsChange{
				"a/go.mod": {`modul moda.com

require gopls.test v0.0.0-goplsworkspace
replace gopls.test => ../../gopls.test2`, false},
			},
			wantChanged: true,
			wantReload:  false,
			finalState: wsState{
				modules: []string{"a/go.mod", "b/go.mod"},
				source:  fileSystemWorkspace,
				// finalDirs should be unchanged: we should preserve dirs in the presence
				// of a broken modfile.
				dirs: []string{".", "a", "b", "../gopls.test"},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			ctx := context.Background()
			dir, err := fake.Tempdir(fake.UnpackTxt(test.initial))
			if err != nil {
				t.Fatal(err)
			}
			defer os.RemoveAll(dir)
			root := span.URIFromPath(dir)

			fs := &osFileSource{}
			excludeNothing := func(string) bool { return false }
			w, err := newWorkspace(ctx, root, "", fs, excludeNothing, false, !test.legacyMode)
			if err != nil {
				t.Fatal(err)
			}
			rel := fake.RelativeTo(dir)
			checkState(ctx, t, fs, rel, w, test.initialState)

			// Apply updates.
			if test.updates != nil {
				changes := make(map[span.URI]*fileChange)
				for k, v := range test.updates {
					content := strings.ReplaceAll(v.content, "$SANDBOX_WORKDIR", string(rel))
					uri := span.URIFromPath(rel.AbsPath(k))
					changes[uri], err = fs.change(ctx, uri, content, v.saved)
					if err != nil {
						t.Fatal(err)
					}
				}
				got, gotReinit := w.Clone(ctx, changes, fs)
				gotChanged := got != w
				if gotChanged != test.wantChanged {
					t.Errorf("w.invalidate(): got changed %t, want %t", gotChanged, test.wantChanged)
				}
				if gotReinit != test.wantReload {
					t.Errorf("w.invalidate(): got reload %t, want %t", gotReinit, test.wantReload)
				}
				checkState(ctx, t, fs, rel, got, test.finalState)
			}
		})
	}
}

func workspaceFromTxtar(t *testing.T, files string) (*workspace, func(), error) {
	ctx := context.Background()
	dir, err := fake.Tempdir(fake.UnpackTxt(files))
	if err != nil {
		return nil, func() {}, err
	}
	cleanup := func() {
		os.RemoveAll(dir)
	}
	root := span.URIFromPath(dir)

	fs := &osFileSource{}
	excludeNothing := func(string) bool { return false }
	workspace, err := newWorkspace(ctx, root, "", fs, excludeNothing, false, false)
	return workspace, cleanup, err
}

func TestWorkspaceParseError(t *testing.T) {
	w, cleanup, err := workspaceFromTxtar(t, `
-- go.work --
go 1.18

usa ./typo
-- typo/go.mod --
module foo
`)
	defer cleanup()
	if err != nil {
		t.Fatalf("error creating workspace: %v; want no error", err)
	}
	w.buildMu.Lock()
	built, buildErr := w.built, w.buildErr
	w.buildMu.Unlock()
	if !built || buildErr == nil {
		t.Fatalf("built, buildErr: got %v, %v; want true, non-nil", built, buildErr)
	}
	var errList modfile.ErrorList
	if !errors.As(buildErr, &errList) {
		t.Fatalf("expected error to be an errorlist; got %v", buildErr)
	}
	if len(errList) != 1 {
		t.Fatalf("expected errorList to have one element; got %v elements", len(errList))
	}
	parseErr := errList[0]
	if parseErr.Pos.Line != 3 {
		t.Fatalf("expected error to be on line 3; got %v", parseErr.Pos.Line)
	}
}

func TestWorkspaceMissingModFile(t *testing.T) {
	w, cleanup, err := workspaceFromTxtar(t, `
-- go.work --
go 1.18

use ./missing
`)
	defer cleanup()
	if err != nil {
		t.Fatalf("error creating workspace: %v; want no error", err)
	}
	w.buildMu.Lock()
	built, buildErr := w.built, w.buildErr
	w.buildMu.Unlock()
	if !built || buildErr == nil {
		t.Fatalf("built, buildErr: got %v, %v; want true, non-nil", built, buildErr)
	}
}

func checkState(ctx context.Context, t *testing.T, fs source.FileSource, rel fake.RelativeTo, got *workspace, want wsState) {
	t.Helper()
	if got.moduleSource != want.source {
		t.Errorf("module source = %v, want %v", got.moduleSource, want.source)
	}
	modules := make(map[span.URI]struct{})
	for k := range got.ActiveModFiles() {
		modules[k] = struct{}{}
	}
	for _, modPath := range want.modules {
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
	gotDirs := got.dirs(ctx, fs)
	gotM := make(map[span.URI]bool)
	for _, dir := range gotDirs {
		gotM[dir] = true
	}
	for _, dir := range want.dirs {
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
	gotSumBytes, err := got.sumFile(ctx, fs)
	if err != nil {
		t.Fatal(err)
	}
	if gotSum := string(gotSumBytes); gotSum != want.sum {
		t.Errorf("got final sum %q, want %q", gotSum, want.sum)
	}
}
