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

// Severities of an LSP diagnostic.
const (
	severityError   = 1
	severityWarning = 2
)

// publishAll publishes the diagnostics of every open document, and clears
// the ones published for documents since closed.
func (p *Proxy) publishAll() {
	p.mu.Lock()
	uris := make(map[string]bool, len(p.docs)+len(p.sent))
	for _, doc := range p.docs {
		uris[doc.uri] = true
	}
	for uri := range p.sent {
		uris[uri] = true
	}
	p.mu.Unlock()
	for _, uri := range slices.Sorted(maps.Keys(uris)) {
		p.publish(uri)
	}
}

// publish sends the merged diagnostics of the client document at uri,
// unless they equal what it sent last. A document with nothing to report
// that never had anything sent gets no message.
func (p *Proxy) publish(uri string) {
	p.pub.Lock()
	defer p.pub.Unlock()
	params, count := p.diagnosticsParams(uri)
	raw, err := encode(params)
	if err != nil {
		p.logf("encoding diagnostics of %s: %v", uri, err)
		return
	}
	p.mu.Lock()
	prev, seen := p.sent[uri]
	same := prev == string(raw) || (count == 0 && !seen)
	if count == 0 && p.docs[uriPathOrEmpty(uri)] == nil {
		delete(p.sent, uri)
	} else {
		p.sent[uri] = string(raw)
	}
	p.mu.Unlock()
	if !same {
		p.send(p.client, &message{Method: "textDocument/publishDiagnostics", Params: raw})
	}
}

// uriPathOrEmpty returns the path of a file URI, or "" for a URI that is
// not one.
func uriPathOrEmpty(uri string) string {
	path, err := uriPath(uri)
	if err != nil {
		return ""
	}
	return path
}

// diagnosticsParams builds the publishDiagnostics params of the client
// document at uri and counts its diagnostics. It merges what gopls
// published for the document, mapped to the source, with the compiler
// diagnostics of the package, and drops a compiler diagnostic that gopls
// reported at the same place with the same text.
func (p *Proxy) diagnosticsParams(uri string) (map[string]any, int) {
	path := uriPathOrEmpty(uri)
	tb := p.tables.Load()
	p.mu.Lock()
	doc := p.docs[path]
	lw := tb.bySrc[path]
	items := p.serverItems(tb, uri, path, lw, doc)
	extra := p.compilerItems(path, doc)
	p.mu.Unlock()
	items = mergeDiagnostics(items, extra)
	params := map[string]any{"uri": uri, "diagnostics": items}
	if doc != nil {
		params["version"] = doc.version
	}
	return params, len(items)
}

// serverItems returns the diagnostics gopls published for the document, in
// the coordinates of the client's text. A loGo/s document shows them only
// when it is open, its lowering is current, and gopls published them for
// the lowered text it holds now. The caller holds p.mu.
func (p *Proxy) serverItems(tb *tables, uri, path string, lw *lowered, doc *document) []any {
	if !isLogosPath(path) {
		return decodeItems(p.srvDiags[uri].raw, tb, nil)
	}
	if doc == nil || lw == nil || !lw.fresh {
		return nil
	}
	sd := p.srvDiags[lw.virtURI()]
	vd := p.virt[lw.virt]
	if vd == nil || (sd.hasVer && sd.version != vd.version) {
		return nil
	}
	return decodeItems(sd.raw, tb, lw)
}

// decodeItems decodes a diagnostics array and maps it to the client's
// coordinates through lw, the lowering of the document it belongs to.
func decodeItems(raw json.RawMessage, tb *tables, lw *lowered) []any {
	v, err := decode(raw)
	if err != nil {
		return nil
	}
	list, _ := tb.toSource().rewrite(v, lw).([]any)
	return list
}

// compilerItems returns the compiler diagnostics of the open document at
// path as LSP diagnostics. Diagnostics computed against another version of
// the document are left out until the next compile replaces them. The
// caller holds p.mu.
func (p *Proxy) compilerItems(path string, doc *document) []any {
	u := p.unitFor(path)
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

// lspDiagnostic converts a compiler diagnostic of the file at path to an
// LSP diagnostic over text t. It reports false for a diagnostic of another
// file, or one whose span lies outside the text.
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

// severityOf returns the LSP severity of a compiler diagnostic severity.
func severityOf(s string) int {
	if s == "warn" {
		return severityWarning
	}
	return severityError
}

// mergeDiagnostics appends the diagnostics of extra that items does not
// already hold. Two diagnostics are the same when they start at the same
// position with the same message.
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

// diagnosticKey identifies a diagnostic by its start and message.
func diagnosticKey(item any) string {
	m, _ := item.(map[string]any)
	msg, _ := m["message"].(string)
	r, _ := m["range"].(map[string]any)
	start, _ := r["start"].(map[string]any)
	p, _ := asPosition(start)
	return fmt.Sprintf("%s\x00%d:%d", msg, p.Line, p.Character)
}
