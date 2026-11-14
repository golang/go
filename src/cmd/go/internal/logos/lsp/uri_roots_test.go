// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
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

func initProxy(t *testing.T, root string) *Proxy {
	t.Helper()
	p := newProxy(discardWriter{}, Options{})
	p.server = &writer{w: discardWriter{}}
	raw, err := encode(map[string]any{"rootUri": pathURI(root)})
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
