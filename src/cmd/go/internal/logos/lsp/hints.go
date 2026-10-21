// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

// HintsFunc serves compiler hints (M21-S2). The proxy calls it for every
// textDocument/inlayHint and textDocument/hover request, after it has
// forwarded the request to gopls, with the method, the params exactly as
// the client sent them (positions in the client's own text, the .lgo or
// .go source it opened), and the hints JSON of the package the document
// belongs to (see HintsJSONFunc). It returns what the hints add to the
// answer: for inlayHint a JSON array of InlayHint, for hover a Hover or a
// JSON null. The proxy appends the array to the inlay hints of gopls and
// appends the contents of the Hover to the hover of gopls. An error leaves
// the answer of gopls as it is.
type HintsFunc func(method string, params, hints json.RawMessage) (json.RawMessage, error)

// HintsJSONFunc returns the hints JSON of the package in dir, the text
// the compiler writes under -logos.mode=hints. buffers holds the unsaved
// text of the open documents by absolute path. The proxy keeps the result
// for a package until the package is compiled again.
type HintsJSONFunc func(ctx context.Context, dir string, buffers map[string][]byte) (json.RawMessage, error)

// CodeActionsFunc turns the compiler autofix candidates into code actions
// (M21-S2). The proxy calls it for every textDocument/codeAction request
// with the compiler diagnostics of the document, each as the JSON line the
// compiler wrote (SPEC Sec 16), the document's URI, its current text, and
// range of the request as LSP Range JSON in the client's own text. It
// returns a JSON array of CodeAction, which the proxy appends to the
// actions of gopls, or answers on its own when the document has no
// current lowering to ask gopls with. An error leaves the answer of gopls
// as it is.
type CodeActionsFunc func(diags [][]byte, uri string, src []byte, rng json.RawMessage) (json.RawMessage, error)

// servesHints reports whether the hints hook answers method.
func servesHints(method string) bool {
	return method == "textDocument/inlayHint" || method == "textDocument/hover"
}

// spawnHints merges the hints into result and answers request id from a
// task the proxy owns, so the compile behind the hints cannot hold up the
// loop that reads gopls. It reports false, having started nothing, once
// the proxy is stopping.
func (p *Proxy) spawnHints(ctx context.Context, id json.RawMessage, pend pending, result any) bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.stopped {
		return false
	}
	p.tasks.Go(func() {
		p.respond(id, p.mergeHints(ctx, pend, result))
		p.finish(id)
	})
	return true
}

// mergeHints returns result with the hints of the hook added. Any failure
// leaves result alone.
func (p *Proxy) mergeHints(ctx context.Context, pend pending, result any) any {
	hints, err := p.packageHints(ctx, pend.path)
	if err != nil {
		p.logf("hints for %s: %v", pend.path, err)
		return result
	}
	raw, err := p.opts.Hints(pend.method, pend.params, hints)
	if err != nil {
		p.logf("hints for %s: %v", pend.path, err)
		return result
	}
	extra, err := decode(raw)
	if err != nil {
		p.logf("hints for %s: %v", pend.path, err)
		return result
	}
	if pend.method == "textDocument/inlayHint" {
		return appendList(result, extra)
	}
	return mergeHover(result, extra)
}

// packageHints returns the hints JSON of the package that holds path,
// from the cache or from the HintsJSON hook.
func (p *Proxy) packageHints(ctx context.Context, path string) (json.RawMessage, error) {
	if p.opts.HintsJSON == nil {
		return nil, errors.New("no hints runner")
	}
	st := p.hintsState(path)
	if st.unit == nil {
		return nil, fmt.Errorf("no open package for %s", path)
	}
	if st.hit {
		return st.hints, nil
	}
	hints, err := p.opts.HintsJSON(ctx, st.unit.dir, st.buffers)
	if err != nil {
		return nil, err
	}
	p.storeHints(st, hints)
	return hints, nil
}

// hintsSnapshot is what packageHints knows about a package at one moment:
// its unit, the generation of the unit's text, and either the cached hints
// JSON or the texts of the open documents to compute it from.
type hintsSnapshot struct {
	unit    *unit
	gen     int
	hit     bool
	hints   json.RawMessage
	buffers map[string][]byte
}

// hintsState returns the hints snapshot of the package that holds path.
func (p *Proxy) hintsState(path string) hintsSnapshot {
	p.mu.Lock()
	defer p.mu.Unlock()
	u := p.unitFor(path)
	if u == nil {
		return hintsSnapshot{}
	}
	if u.hintsOK {
		return hintsSnapshot{unit: u, gen: u.gen, hit: true, hints: u.hints}
	}
	return hintsSnapshot{unit: u, gen: u.gen, buffers: p.buffersLocked()}
}

// storeHints caches hints computed from the snapshot st, unless the unit
// changed or was compiled again while they were computed.
func (p *Proxy) storeHints(st hintsSnapshot, hints json.RawMessage) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.units[st.unit.dir] == st.unit && st.unit.gen == st.gen {
		st.unit.hints, st.unit.hintsOK = hints, true
	}
}

// buffersLocked returns the texts of the open documents by path. The
// caller holds p.mu.
func (p *Proxy) buffersLocked() map[string][]byte {
	buffers := make(map[string][]byte, len(p.docs))
	for path, doc := range p.docs {
		buffers[path] = doc.text.data
	}
	return buffers
}

// appendList appends the list extra to the list base. A base that is not
// a list, an absent answer of gopls, counts as empty.
func appendList(base, extra any) any {
	b, _ := base.([]any)
	e, _ := extra.([]any)
	if b == nil && e == nil {
		return base
	}
	return append(b, e...)
}

// hoverSeparator joins the hover text of gopls and the hover text of the
// hints.
const hoverSeparator = "\n\n---\n\n"

// mergeHover joins two decoded hovers. When either is absent the other is
// the answer. When both carry markup of one kind, their text is joined
// under the range of the first; when the kinds differ, the first stands.
func mergeHover(base, extra any) any {
	b, _ := base.(map[string]any)
	e, _ := extra.(map[string]any)
	if b == nil {
		return extra
	}
	if e == nil {
		return base
	}
	bc, _ := b["contents"].(map[string]any)
	ec, _ := e["contents"].(map[string]any)
	bk, _ := bc["kind"].(string)
	ek, _ := ec["kind"].(string)
	bv, ok1 := bc["value"].(string)
	ev, ok2 := ec["value"].(string)
	if bk == "" || bk != ek || !ok1 || !ok2 {
		return base
	}
	bc["value"] = strings.Join([]string{bv, ev}, hoverSeparator)
	return base
}

// mergeCodeActions appends the hook's code actions for the request of
// pend to result. When an edit crossed the request in flight, the whole
// answer is dropped: a code action applies its edits to the current text,
// so neither the compiler fixes for the old range nor the gopls actions in
// the old coordinates may cross it. The client re-requests after the edit.
func (p *Proxy) mergeCodeActions(pend pending, result any) any {
	if p.opts.CodeActions == nil {
		return result
	}
	var lines [][]byte
	var src []byte
	if pend.fixHas {
		var moved bool
		lines, src, moved = p.autofixSince(pend.path, pend.fixVersion)
		if moved {
			return []any{}
		}
	} else {
		lines, src = p.autofixCurrent(pend.path)
	}
	if src == nil {
		return result
	}
	var req struct {
		Range json.RawMessage `json:"range"`
	}
	if err := json.Unmarshal(pend.params, &req); err != nil {
		return result
	}
	raw, err := p.opts.CodeActions(lines, pathURI(pend.path), src, req.Range)
	if err != nil {
		p.logf("code actions for %s: %v", pend.path, err)
		return result
	}
	extra, err := decode(raw)
	if err != nil {
		p.logf("code actions for %s: %v", pend.path, err)
		return result
	}
	return appendList(result, extra)
}

// docVersion returns the version of the open document at path, or false
// when the document is not open.
func (p *Proxy) docVersion(path string) (int, bool) {
	p.mu.Lock()
	defer p.mu.Unlock()
	doc := p.docs[path]
	if doc == nil {
		return 0, false
	}
	return doc.version, true
}

// autofixState is one locked observation of a document's autofix inputs:
// its version, whether it is open, and, when its diagnostics describe
// that version, their lines and its text.
type autofixState struct {
	lines   [][]byte
	src     []byte
	version int
	open    bool
	current bool
}

// autofixObserve captures the autofix inputs of the document at path
// under one lock, so an edit cannot slip between the version check and
// the read.
func (p *Proxy) autofixObserve(path string) autofixState {
	p.mu.Lock()
	defer p.mu.Unlock()
	doc, u := p.docs[path], p.unitFor(path)
	if doc == nil {
		return autofixState{}
	}
	st := autofixState{version: doc.version, open: true}
	if u == nil || u.diagVersions[path] != doc.version {
		return st
	}
	for _, d := range u.diags {
		if d.File == path {
			st.lines = append(st.lines, d.Raw)
		}
	}
	st.src, st.current = doc.text.data, true
	return st
}

// autofixCurrent returns the compiler diagnostic lines and text for an
// answer computed without gopls, or nothing when the document is not
// open with current diagnostics.
func (p *Proxy) autofixCurrent(path string) ([][]byte, []byte) {
	st := p.autofixObserve(path)
	if !st.current {
		return nil, nil
	}
	return st.lines, st.src
}

// autofixSince returns the compiler diagnostic lines and text for a
// request that saw version. It reports moved when the document has since
// changed or closed: the whole codeAction answer must be dropped, since
// neither the old range nor the old coordinates apply. A document that is
// merely behind on diagnostics is not a move: there is nothing
// compiler-side to add, and the gopls answer, if any, stands.
func (p *Proxy) autofixSince(path string, version int) (lines [][]byte, src []byte, moved bool) {
	st := p.autofixObserve(path)
	if !st.open || st.version != version {
		return nil, nil, true
	}
	if !st.current {
		return nil, nil, false
	}
	return st.lines, st.src, false
}
