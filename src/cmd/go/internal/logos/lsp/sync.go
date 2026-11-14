// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
)

type textDocumentItem struct {
	URI        string `json:"uri"`
	LanguageID string `json:"languageId"`
	Version    int    `json:"version"`
	Text       string `json:"text"`
}

type didOpenParams struct {
	TextDocument textDocumentItem `json:"textDocument"`
}

type contentChange struct {
	Range *Range `json:"range,omitempty"`
	Text  string `json:"text"`
}

type docIdentifier struct {
	URI string `json:"uri"`
}

type versionedID struct {
	URI     string `json:"uri"`
	Version int    `json:"version"`
}

type didChangeParams struct {
	TextDocument   versionedID     `json:"textDocument"`
	ContentChanges []contentChange `json:"contentChanges"`
}

type docParams struct {
	TextDocument docIdentifier `json:"textDocument"`
}

type watchedParams struct {
	Changes []struct {
		URI string `json:"uri"`
	} `json:"changes"`
}

func (p *Proxy) fromClient(ctx context.Context, m *message) {
	switch {
	case m.isResponse():
		p.send(p.server, m)
	case m.isRequest():
		p.clientRequest(ctx, m)
	default:
		p.clientNotification(ctx, m)
	}
}

func (p *Proxy) clientNotification(ctx context.Context, m *message) {
	switch m.Method {
	case "textDocument/didOpen":
		p.didOpen(ctx, m)
	case "textDocument/didChange":
		p.didChange(ctx, m)
	case "textDocument/didSave":
		p.didSave(ctx, m)
	case "textDocument/didClose":
		p.didClose(ctx, m)
	case "workspace/didChangeWatchedFiles":
		p.watched(ctx, m)
	case "workspace/didChangeWorkspaceFolders":
		p.changeWorkspaceFolders(m.Params)
		p.send(p.server, m)
	case "$/cancelRequest":
		p.cancelPending(m)
		p.send(p.server, m)
	default:
		p.send(p.server, m)
	}
}

func (p *Proxy) cancelPending(m *message) {
	v, err := decode(m.Params)
	root, _ := v.(map[string]any)
	if err != nil || root == nil {
		return
	}
	raw, err := encode(root["id"])
	if err != nil {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	delete(p.pending, string(raw))
}

func trackedPath(path string) bool { return isLogosPath(path) || isGoPath(path) }

func (p *Proxy) docPath(uri string) string {
	path, err := uriPath(uri)
	if err != nil || !trackedPath(path) {
		return ""
	}
	return p.contain(path)
}

func (p *Proxy) contain(path string) string {
	p.rootsMu.RLock()
	defer p.rootsMu.RUnlock()
	if len(p.roots) == 0 {
		return path
	}
	for _, root := range p.roots {
		prefix := root
		if !strings.HasSuffix(prefix, string(os.PathSeparator)) {
			prefix += string(os.PathSeparator)
		}
		if path == root || strings.HasPrefix(path, prefix) {
			return path
		}
	}
	return ""
}

func (p *Proxy) rememberRoots(params any) {
	root, _ := params.(map[string]any)
	var uris []string
	if uri, _ := root["rootUri"].(string); uri != "" {
		uris = append(uris, uri)
	}
	folders, _ := root["workspaceFolders"].([]any)
	for _, f := range folders {
		if m, _ := f.(map[string]any); m != nil {
			if uri, _ := m["uri"].(string); uri != "" {
				uris = append(uris, uri)
			}
		}
	}
	var roots []string
	for _, uri := range uris {
		if path, err := uriPath(uri); err == nil && path != "" {
			roots = append(roots, path)
		}
	}
	p.rootsMu.Lock()
	p.roots = roots
	p.rootsMu.Unlock()
}

func (p *Proxy) changeWorkspaceFolders(raw json.RawMessage) {
	type folder struct {
		URI string `json:"uri"`
	}
	var params struct {
		Event struct {
			Added   []folder `json:"added"`
			Removed []folder `json:"removed"`
		} `json:"event"`
	}
	if err := json.Unmarshal(raw, &params); err != nil {
		p.logf("didChangeWorkspaceFolders: %v", err)
		return
	}
	p.rootsMu.Lock()
	defer p.rootsMu.Unlock()
	for _, f := range params.Event.Removed {
		if path, err := uriPath(f.URI); err == nil && path != "" {
			p.roots = slices.DeleteFunc(p.roots, func(root string) bool { return root == path })
		}
	}
	for _, f := range params.Event.Added {
		if path, err := uriPath(f.URI); err == nil && path != "" && !slices.Contains(p.roots, path) {
			p.roots = append(p.roots, path)
		}
	}
}

func (p *Proxy) forwardUnlessLogos(m *message, path string) {
	if !isLogosPath(path) {
		p.send(p.server, m)
	}
}

func (p *Proxy) didOpen(ctx context.Context, m *message) {
	var params didOpenParams
	if err := json.Unmarshal(m.Params, &params); err != nil {
		p.logf("didOpen: %v", err)
		p.send(p.server, m)
		return
	}
	item := params.TextDocument
	path := p.docPath(item.URI)
	if path != "" {
		if len(item.Text) > maxDocBytes {
			p.logf("didOpen: ignoring oversized document %s", path)
		} else {
			p.st.setDoc(item.URI, path, item.Version, newText([]byte(item.Text)))
		}
	}
	p.forwardUnlessLogos(m, path)
}

func (p *Proxy) didChange(ctx context.Context, m *message) {
	var params didChangeParams
	if err := json.Unmarshal(m.Params, &params); err != nil {
		p.logf("didChange: %v", err)
		p.send(p.server, m)
		return
	}
	path := p.docPath(params.TextDocument.URI)
	if path != "" && !p.st.change(path, params.TextDocument.Version, params.ContentChanges) {
		p.logf("didChange: ignoring change for %s version %d", path, params.TextDocument.Version)
		return
	}
	p.forwardUnlessLogos(m, path)
}

func applyChanges(t *text, changes []contentChange) *text {
	for _, c := range changes {
		if c.Range == nil {
			t = newText([]byte(c.Text))
			continue
		}
		start, end := t.offset(c.Range.Start), t.offset(c.Range.End)
		if start > end {
			continue
		}
		t = t.replace(start, end, c.Text)
	}
	return t
}

func (p *Proxy) didSave(ctx context.Context, m *message) {
	var params docParams
	if err := json.Unmarshal(m.Params, &params); err != nil {
		p.logf("didSave: %v", err)
		p.send(p.server, m)
		return
	}
	path := p.docPath(params.TextDocument.URI)
	if p.st.open(path) {
		p.ensure(ctx)
	}
	p.forwardUnlessLogos(m, path)
}

func (p *Proxy) didClose(ctx context.Context, m *message) {
	var params docParams
	if err := json.Unmarshal(m.Params, &params); err != nil {
		p.logf("didClose: %v", err)
		p.send(p.server, m)
		return
	}
	path := p.docPath(params.TextDocument.URI)
	retained := p.st.forget(path)
	p.mu.Lock()
	delete(p.srvDiags, params.TextDocument.URI)
	p.mu.Unlock()
	p.forwardUnlessLogos(m, path)
	if retained {
		p.ensure(ctx)
		return
	}
	p.st.mu.Lock()
	need := p.st.unitNeed()
	p.st.mu.Unlock()
	p.work.Lock()
	defer p.work.Unlock()
	p.mu.Lock()
	p.pruneLocked(need)
	p.mu.Unlock()
	p.syncGopls()
	p.publishAll()
}

func (p *Proxy) watched(ctx context.Context, m *message) {
	var params watchedParams
	if err := json.Unmarshal(m.Params, &params); err == nil {
		p.markIfRelevant(params)
	}
	p.send(p.server, m)
}

func (p *Proxy) markIfRelevant(params watchedParams) {
	relevant := false
	for _, c := range params.Changes {
		path, err := uriPath(c.URI)
		if err == nil && (trackedPath(path) || isModuleFile(path)) {
			relevant = true
		}
	}
	if !relevant {
		return
	}
	p.st.mu.Lock()
	defer p.st.mu.Unlock()
	for _, u := range p.st.units {
		u.dirty = true
		u.gen++
	}
	if len(p.st.units) > 0 {
		p.opts.Compiler.DiskChanged()
	}
}

func isModuleFile(path string) bool {
	switch filepath.Base(path) {
	case "go.mod", "go.sum", "go.work", "go.work.sum", "logos.toml":
		return true
	}
	return false
}
