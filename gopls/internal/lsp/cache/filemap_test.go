// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/gopls/internal/span"
)

func TestFileMap(t *testing.T) {
	const (
		set = iota
		del
	)
	type op struct {
		op      int // set or remove
		path    string
		overlay bool
	}
	tests := []struct {
		label        string
		ops          []op
		wantFiles    []string
		wantOverlays []string
		wantDirs     []string
	}{
		{"empty", nil, nil, nil, nil},
		{"singleton", []op{
			{set, "/a/b", false},
		}, []string{"/a/b"}, nil, []string{"/", "/a"}},
		{"overlay", []op{
			{set, "/a/b", true},
		}, []string{"/a/b"}, []string{"/a/b"}, []string{"/", "/a"}},
		{"replace overlay", []op{
			{set, "/a/b", true},
			{set, "/a/b", false},
		}, []string{"/a/b"}, nil, []string{"/", "/a"}},
		{"multi dir", []op{
			{set, "/a/b", false},
			{set, "/c/d", false},
		}, []string{"/a/b", "/c/d"}, nil, []string{"/", "/a", "/c"}},
		{"empty dir", []op{
			{set, "/a/b", false},
			{set, "/c/d", false},
			{del, "/a/b", false},
		}, []string{"/c/d"}, nil, []string{"/", "/c"}},
	}

	// Normalize paths for windows compatibility.
	normalize := func(path string) string {
		return strings.TrimPrefix(filepath.ToSlash(path), "C:") // the span packages adds 'C:'
	}

	for _, test := range tests {
		t.Run(test.label, func(t *testing.T) {
			m := newFileMap()
			for _, op := range test.ops {
				uri := span.URIFromPath(filepath.FromSlash(op.path))
				switch op.op {
				case set:
					var fh source.FileHandle
					if op.overlay {
						fh = &Overlay{uri: uri}
					} else {
						fh = &DiskFile{uri: uri}
					}
					m.Set(uri, fh)
				case del:
					m.Delete(uri)
				}
			}

			var gotFiles []string
			m.Range(func(uri span.URI, _ source.FileHandle) {
				gotFiles = append(gotFiles, normalize(uri.Filename()))
			})
			sort.Strings(gotFiles)
			if diff := cmp.Diff(test.wantFiles, gotFiles); diff != "" {
				t.Errorf("Files mismatch (-want +got):\n%s", diff)
			}

			var gotOverlays []string
			for _, o := range m.Overlays() {
				gotOverlays = append(gotOverlays, normalize(o.URI().Filename()))
			}
			if diff := cmp.Diff(test.wantOverlays, gotOverlays); diff != "" {
				t.Errorf("Overlays mismatch (-want +got):\n%s", diff)
			}

			var gotDirs []string
			m.Dirs().Range(func(dir string) {
				gotDirs = append(gotDirs, normalize(dir))
			})
			sort.Strings(gotDirs)
			if diff := cmp.Diff(test.wantDirs, gotDirs); diff != "" {
				t.Errorf("Dirs mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
