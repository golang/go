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

type HintsFunc func(method string, params, hints json.RawMessage) (json.RawMessage, error)

type HintsJSONFunc func(ctx context.Context, dir string, buffers map[string][]byte) (json.RawMessage, error)

type CodeActionsFunc func(diags [][]byte, uri string, src []byte, rng json.RawMessage) (json.RawMessage, error)

func servesHints(method string) bool {
	return method == "textDocument/inlayHint" || method == "textDocument/hover"
}

func (p *Proxy) spawnHints(ctx context.Context, id json.RawMessage, pend pending, result any) bool {
	p.stopMu.Lock()
	defer p.stopMu.Unlock()
	if p.stopped {
		return false
	}
	select {
	case p.hintSem <- struct{}{}:
	default:
		return false
	}
	p.tasks.Go(func() {
		defer func() { <-p.hintSem }()
		p.respond(id, p.mergeHints(ctx, pend, result))
		p.finish(id)
	})
	return true
}

func (p *Proxy) mergeHints(ctx context.Context, pend pending, result any) any {
	hints, st, err := p.packageHints(ctx, pend.path)
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
	if !p.hintsFresh(st) {
		return result
	}
	if pend.method == "textDocument/inlayHint" {
		return appendList(result, extra)
	}
	return mergeHover(result, extra)
}

func (p *Proxy) packageHints(ctx context.Context, path string) (json.RawMessage, hintsSnapshot, error) {
	if p.opts.HintsJSON == nil {
		return nil, hintsSnapshot{}, errors.New("no hints runner")
	}
	st := p.hintsState(path)
	if st.unit == nil {
		return nil, st, fmt.Errorf("no open package for %s", path)
	}
	if st.hit {
		return st.hints, st, nil
	}
	hints, err := p.opts.HintsJSON(ctx, st.unit.dir, st.buffers)
	if err != nil {
		return nil, st, err
	}
	p.storeHints(st, hints)
	return hints, st, nil
}

type hintsSnapshot struct {
	unit     *unit
	gen      int
	hit      bool
	hints    json.RawMessage
	buffers  map[string][]byte
	versions map[string]int
}

func (p *Proxy) hintsState(path string) hintsSnapshot {
	p.st.mu.Lock()
	defer p.st.mu.Unlock()
	u := p.st.unitFor(path)
	if u == nil {
		return hintsSnapshot{}
	}
	buffers := make(map[string][]byte, len(p.st.docs))
	versions := make(map[string]int, len(p.st.docs))
	for path, doc := range p.st.docs {
		buffers[path] = doc.text.data
		versions[path] = doc.version
	}
	if u.hintsOK {
		return hintsSnapshot{unit: u, gen: u.gen, hit: true, hints: u.hints, buffers: buffers, versions: versions}
	}
	return hintsSnapshot{unit: u, gen: u.gen, buffers: buffers, versions: versions}
}

func (p *Proxy) hintsFresh(st hintsSnapshot) bool {
	p.st.mu.Lock()
	defer p.st.mu.Unlock()
	if st.unit == nil || p.st.units[st.unit.dir] != st.unit || st.unit.gen != st.gen {
		return false
	}
	return versionsEqualLocked(p.st.docs, st.versions)
}

func versionsEqualLocked(docs map[string]*document, versions map[string]int) bool {
	if len(versions) != len(docs) {
		return false
	}
	for path, version := range versions {
		doc := docs[path]
		if doc == nil || doc.version != version {
			return false
		}
	}
	return true
}

func (p *Proxy) storeHints(st hintsSnapshot, hints json.RawMessage) {
	p.st.mu.Lock()
	defer p.st.mu.Unlock()
	if st.unit == nil || p.st.units[st.unit.dir] != st.unit || st.unit.gen != st.gen {
		return
	}
	if !versionsEqualLocked(p.st.docs, st.versions) {
		return
	}
	st.unit.hints, st.unit.hintsOK = hints, true
}

func appendList(base, extra any) any {
	b, _ := base.([]any)
	e, _ := extra.([]any)
	if b == nil && e == nil {
		return base
	}
	return append(b, e...)
}

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
	bc["value"] = strings.Join([]string{bv, ev}, "\n\n---\n\n")
	return base
}

func (p *Proxy) mergeCodeActions(pend pending, result any) any {
	if p.opts.CodeActions == nil {
		return result
	}
	st := p.autofixObserve(pend.path)
	if pend.fixHas && (!st.open || st.version != pend.fixVersion) {
		return []any{}
	}
	lines, src := st.lines, st.src
	if !st.current {
		lines, src = nil, nil
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

func (p *Proxy) docVersion(path string) (int, bool) {
	p.st.mu.Lock()
	defer p.st.mu.Unlock()
	doc := p.st.docs[path]
	if doc == nil {
		return 0, false
	}
	return doc.version, true
}

type autofixState struct {
	lines   [][]byte
	src     []byte
	version int
	open    bool
	current bool
}

func (p *Proxy) autofixObserve(path string) autofixState {
	p.st.mu.Lock()
	defer p.st.mu.Unlock()
	doc, u := p.st.docs[path], p.st.unitFor(path)
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
