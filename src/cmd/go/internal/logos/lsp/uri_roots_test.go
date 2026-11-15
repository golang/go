// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"os"
	"path/filepath"
	"testing"
)

func TestURIAuthorityRejected(t *testing.T) {
	if path, err := uriPath("file://evil.com/etc/passwd.lgo"); err == nil {
		t.Fatalf("uriPath mapped a remote authority to %q, want an error", path)
	}
	if _, err := uriPath("https://example.com/x.lgo"); err == nil {
		t.Fatal("uriPath accepted a non-file scheme, want an error")
	}
	if path, err := uriPath("file://localhost/etc/x.lgo"); err != nil || path == "" {
		t.Fatalf("uriPath rejected localhost: %q, %v", path, err)
	}
	if path, err := uriPath("file://LOCALHOST/etc/x.lgo"); err != nil || path == "" {
		t.Fatalf("uriPath rejected uppercase localhost: %q, %v", path, err)
	}
}

func initProxy(t *testing.T, root string, folders ...string) *Proxy {
	t.Helper()
	p := newProxy(discardWriter{}, Options{})
	p.server = &writer{w: discardWriter{}}
	params := map[string]any{"rootUri": pathURI(root)}
	if len(folders) > 0 {
		var workspaceFolders []map[string]string
		for _, folder := range folders {
			workspaceFolders = append(workspaceFolders, map[string]string{"uri": pathURI(folder), "name": filepath.Base(folder)})
		}
		params["workspaceFolders"] = workspaceFolders
	}
	raw, err := encode(params)
	if err != nil {
		t.Fatal(err)
	}
	p.initialize(&message{ID: []byte("1"), Method: "initialize", Params: raw})
	return p
}

type discardWriter struct{}

func (discardWriter) Write(b []byte) (int, error) { return len(b), nil }

func TestOutsideRootsNotTracked(t *testing.T) {
	root := t.TempDir()
	p := initProxy(t, root)
	if got := p.docPath(pathURI(root + "/a.lgo")); got == "" {
		t.Fatal("docPath dropped a URI inside the workspace roots")
	}
	if got := p.docPath("file://evil.com/etc/passwd.lgo"); got != "" {
		t.Fatalf("docPath tracked a remote authority: %q", got)
	}
	if got := p.docPath("file:///etc/evil.lgo"); got != "" {
		t.Fatalf("docPath tracked a path outside the roots: %q", got)
	}
	if got := p.docPath("file://" + root + "/%2e%2e/etc/x.lgo"); got != "" {
		t.Fatalf("docPath tracked an escaped path: %q", got)
	}
}

func TestRootSeparatorsTrackDocuments(t *testing.T) {
	dir := t.TempDir()
	root := filepath.Join(dir, "work")
	for _, tt := range []struct {
		name string
		root string
		path string
		want bool
	}{
		{
			name: "filesystem root",
			root: filepath.VolumeName(dir) + string(os.PathSeparator),
			path: filepath.Join(dir, "a.lgo"),
			want: true,
		},
		{
			name: "trailing separator",
			root: root + "/",
			path: filepath.Join(root, "a.lgo"),
			want: true,
		},
		{
			name: "sibling prefix",
			root: root,
			path: filepath.Join(dir, "workspace", "a.lgo"),
			want: false,
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			p := initProxy(t, tt.root)
			uri := pathURI(tt.path)
			raw, err := encode(didOpenParams{TextDocument: textDocumentItem{
				URI: uri, LanguageID: "logos", Version: 1, Text: "package p\n",
			}})
			if err != nil {
				t.Fatal(err)
			}
			p.clientNotification(t.Context(), &message{Method: "textDocument/didOpen", Params: raw})
			if got := p.st.open(tt.path); got != tt.want {
				t.Errorf("open(%q) = %v, want %v", tt.path, got, tt.want)
			}
		})
	}
}

func TestWorkspaceFolderChangesUpdateTrackedRoots(t *testing.T) {
	root := t.TempDir()
	added := t.TempDir()
	p := initProxy(t, root)
	notify := func(method string, params any) {
		t.Helper()
		raw, err := encode(params)
		if err != nil {
			t.Fatal(err)
		}
		p.clientNotification(t.Context(), &message{Method: method, Params: raw})
	}
	open := func(name string, wantTracked bool) {
		t.Helper()
		path := filepath.Join(added, name)
		uri := pathURI(path)
		notify("textDocument/didOpen", didOpenParams{TextDocument: textDocumentItem{
			URI: uri, LanguageID: "logos", Version: 1, Text: "package added\n",
		}})
		if got := p.docPath(uri); (got != "") != wantTracked {
			t.Errorf("docPath(%q) = %q, want tracked = %v", uri, got, wantTracked)
		}
		if got := p.st.open(path); got != wantTracked {
			t.Errorf("open(%q) = %v, want %v", path, got, wantTracked)
		}
	}
	folder := map[string]any{"uri": pathURI(added), "name": "added"}
	open("before.lgo", false)
	notify("workspace/didChangeWorkspaceFolders", map[string]any{
		"event": map[string]any{"added": []any{folder, folder}, "removed": []any{}},
	})
	open("during.lgo", true)
	notify("workspace/didChangeWorkspaceFolders", map[string]any{
		"event": map[string]any{"added": []any{}, "removed": []any{folder}},
	})
	open("after.lgo", false)
	if got := p.docPath(pathURI(filepath.Join(root, "still.lgo"))); got == "" {
		t.Fatal("removing the added folder also removed the initial workspace root")
	}
}

func TestOpenDocumentSurvivesWorkspaceFolderRemoval(t *testing.T) {
	root := t.TempDir()
	removed := t.TempDir()
	p := initProxy(t, root, root, removed)
	notify := func(method string, params any) {
		t.Helper()
		raw, err := encode(params)
		if err != nil {
			t.Fatal(err)
		}
		p.clientNotification(t.Context(), &message{Method: method, Params: raw})
	}
	path := filepath.Join(removed, "open.lgo")
	uri := pathURI(path)
	wantText := "package edited\n"
	if err := os.WriteFile(path, []byte(wantText), 0600); err != nil {
		t.Fatal(err)
	}
	notify("textDocument/didOpen", didOpenParams{TextDocument: textDocumentItem{
		URI: uri, LanguageID: "logos", Version: 1, Text: "package original\n",
	}})
	if !p.st.open(path) {
		t.Fatal("document in the second workspace folder was not tracked")
	}
	notify("workspace/didChangeWorkspaceFolders", map[string]any{
		"event": map[string]any{
			"added":   []any{},
			"removed": []any{map[string]any{"uri": pathURI(removed), "name": "removed"}},
		},
	})
	notify("textDocument/didChange", didChangeParams{
		TextDocument:   versionedID{URI: uri, Version: 2},
		ContentChanges: []contentChange{{Text: wantText}},
	})
	var gotText string
	p.st.mu.Lock()
	if doc := p.st.docs[path]; doc != nil {
		gotText = string(doc.text.data)
	}
	p.st.mu.Unlock()
	if gotText != wantText {
		t.Errorf("document text after folder removal = %q, want %q", gotText, wantText)
	}
	notify("textDocument/didClose", docParams{TextDocument: docIdentifier{URI: uri}})
	if p.st.open(path) {
		t.Error("closed document from the removed workspace folder is still tracked")
	}
	newPath := filepath.Join(removed, "new.lgo")
	notify("textDocument/didOpen", didOpenParams{TextDocument: textDocumentItem{
		URI: pathURI(newPath), LanguageID: "logos", Version: 1, Text: "package newdoc\n",
	}})
	if p.st.open(newPath) {
		t.Error("new document in the removed workspace folder was tracked")
	}
}
