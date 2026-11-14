// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"encoding/json"
	"fmt"
	"maps"
	"slices"
)

const (
	severityError   = 1
	severityWarning = 2
)

func (p *Proxy) publishAll() {
	p.st.mu.Lock()
	uris := make(map[string]bool, len(p.st.docs))
	for _, doc := range p.st.docs {
		uris[doc.uri] = true
	}
	p.st.mu.Unlock()
	p.mu.Lock()
	for uri := range p.sent {
		uris[uri] = true
	}
	p.mu.Unlock()
	for _, uri := range slices.Sorted(maps.Keys(uris)) {
		p.publish(uri)
	}
}

func (p *Proxy) publish(uri string) {
	p.pub.Lock()
	defer p.pub.Unlock()
	params, count := p.diagnosticsParams(uri)
	raw, err := encode(params)
	if err != nil {
		p.logf("encoding diagnostics of %s: %v", uri, err)
		return
	}
	open := false
	if count == 0 {
		open = p.st.open(p.docPath(uri))
	}
	p.mu.Lock()
	prev, seen := p.sent[uri]
	same := prev == string(raw) || (count == 0 && !seen)
	if count == 0 && !open {
		delete(p.sent, uri)
	} else {
		p.sent[uri] = string(raw)
	}
	p.mu.Unlock()
	if !same {
		p.send(p.client, &message{Method: "textDocument/publishDiagnostics", Params: raw})
	}
}

func (p *Proxy) diagnosticsParams(uri string) (map[string]any, int) {
	path := p.docPath(uri)
	p.st.mu.Lock()
	doc := p.st.docs[path]
	u := p.st.unitFor(path)
	extra := p.compilerItems(path, doc, u)
	open, version := doc != nil, 0
	if doc != nil {
		version = doc.version
	}
	p.st.mu.Unlock()
	tb := p.tables.Load()
	p.mu.Lock()
	lw := tb.bySrc[path]
	var items []any
	if open || !isLogosPath(path) {
		items = p.serverItems(tb, uri, path, lw)
	}
	p.mu.Unlock()
	items = mergeDiagnostics(items, extra)
	params := map[string]any{"uri": uri, "diagnostics": items}
	if open {
		params["version"] = version
	}
	return params, len(items)
}

func (p *Proxy) serverItems(tb *tables, uri, path string, lw *lowered) []any {
	if !isLogosPath(path) {
		return decodeItems(p.srvDiags[uri].raw, tb, nil)
	}
	if lw == nil || !lw.fresh {
		return nil
	}
	sd := p.srvDiags[lw.virtURI()]
	vd := p.virt[lw.virt]
	if vd == nil || (sd.hasVer && sd.version != vd.version) {
		return nil
	}
	return decodeItems(sd.raw, tb, lw)
}

func decodeItems(raw json.RawMessage, tb *tables, lw *lowered) []any {
	v, err := decode(raw)
	if err != nil {
		return nil
	}
	list, _ := tb.toSource().rewrite(v, lw).([]any)
	return list
}

func (p *Proxy) compilerItems(path string, doc *document, u *unit) []any {
	if doc == nil || u == nil || u.diagVersions[path] != doc.version {
		return nil
	}
	var items []any
	for _, d := range u.diags {
		if item, ok := lspDiagnostic(d, path, doc.text); ok {
			items = append(items, item)
		}
	}
	return items
}

func lspDiagnostic(d Diagnostic, path string, t *text) (map[string]any, bool) {
	if d.File != path {
		return nil, false
	}
	start, ok := t.lineCol(d.Span.StartLine, d.Span.StartCol)
	if !ok {
		return nil, false
	}
	end, ok := t.lineCol(d.Span.EndLine, d.Span.EndCol)
	end = max(start, end)
	if !ok {
		end = start
	}
	item := map[string]any{
		"range":    rangeJSON(Range{Start: t.position(start), End: t.position(end)}),
		"severity": severityOf(d.Severity),
		"source":   "logos",
		"message":  d.Message,
	}
	if d.RuleID != "" {
		item["code"] = d.RuleID
	}
	return item, true
}

func severityOf(s string) int {
	if s == "warn" {
		return severityWarning
	}
	return severityError
}

func mergeDiagnostics(items, extra []any) []any {
	have := make(map[string]bool, len(items))
	for _, item := range items {
		have[diagnosticKey(item)] = true
	}
	merged := slices.Clone(items)
	for _, item := range extra {
		if !have[diagnosticKey(item)] {
			merged = append(merged, item)
		}
	}
	if merged == nil {
		merged = []any{}
	}
	return merged
}

func diagnosticKey(item any) string {
	m, _ := item.(map[string]any)
	msg, _ := m["message"].(string)
	r, _ := m["range"].(map[string]any)
	start, _ := r["start"].(map[string]any)
	p, _ := asPosition(start)
	return fmt.Sprintf("%s\x00%d:%d", msg, p.Line, p.Character)
}
