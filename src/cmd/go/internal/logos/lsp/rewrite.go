// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"bytes"
	"encoding/json"
	"strconv"
	"strings"
)

type lowered struct {
	src   string
	virt  string
	pm    *posMap
	fresh bool
	gen   int
}

func (lw *lowered) srcURI() string { return pathURI(lw.src) }

func (lw *lowered) virtURI() string { return pathURI(lw.virt) }

type tables struct {
	bySrc  map[string]*lowered
	byVirt map[string]*lowered
}

func (tb *tables) toLowered() *rewriter {
	return &rewriter{
		route: func(uri string) *lowered { return tb.lookup(tb.bySrc, uri) },
		name:  (*lowered).virtURI,
		pos:   func(lw *lowered, p Position) Position { return lw.pm.toLowered(p) },
		rng:   func(lw *lowered, r Range) Range { return lw.pm.rangeToLowered(r) },
	}
}

func (tb *tables) toSource() *rewriter {
	return &rewriter{
		route:       func(uri string) *lowered { return tb.lookup(tb.byVirt, uri) },
		name:        (*lowered).srcURI,
		pos:         func(lw *lowered, p Position) Position { return lw.pm.toOriginal(p) },
		rng:         func(lw *lowered, r Range) Range { return lw.pm.rangeToOriginal(r) },
		dropVersion: true,
	}
}

func (tb *tables) lookup(m map[string]*lowered, uri string) *lowered {
	path, err := uriPath(uri)
	if err != nil {
		return nil
	}
	return m[path]
}

type rewriter struct {
	route       func(uri string) *lowered
	name        func(lw *lowered) string
	pos         func(lw *lowered, p Position) Position
	rng         func(lw *lowered, r Range) Range
	dropVersion bool
}

func (w *rewriter) rewrite(v any, ctx *lowered) any {
	switch t := v.(type) {
	case []any:
		for i := range t {
			t[i] = w.rewrite(t[i], ctx)
		}
	case map[string]any:
		return w.object(t, ctx)
	}
	return v
}

func (w *rewriter) object(m map[string]any, ctx *lowered) any {
	if r, ok := asRange(m); ok {
		if ctx == nil {
			return m
		}
		return rangeJSON(w.rng(ctx, r))
	}
	if p, ok := asPosition(m); ok {
		if ctx == nil {
			return m
		}
		return positionJSON(w.pos(ctx, p))
	}
	own := w.owner(m, ctx)
	w.renameURIs(m)
	for key, v := range m {
		m[key] = w.child(key, v, own, ctx)
	}
	return m
}

func (w *rewriter) owner(m map[string]any, ctx *lowered) *lowered {
	if uri, ok := namedURI(m); ok {
		return w.route(uri)
	}
	return ctx
}

func namedURI(m map[string]any) (string, bool) {
	for _, key := range []string{"uri", "targetUri"} {
		if s, ok := m[key].(string); ok {
			return s, true
		}
	}
	if td, ok := m["textDocument"].(map[string]any); ok {
		s, ok := td["uri"].(string)
		return s, ok
	}
	return "", false
}

func (w *rewriter) renameURIs(m map[string]any) {
	for _, key := range []string{"uri", "targetUri", "oldUri", "newUri"} {
		s, ok := m[key].(string)
		if !ok {
			continue
		}
		lw := w.route(s)
		if lw == nil {
			continue
		}
		m[key] = w.name(lw)
		if _, versioned := m["version"]; key == "uri" && versioned && w.dropVersion {
			m["version"] = nil
		}
	}
}

func (w *rewriter) child(key string, v any, own, ctx *lowered) any {
	switch key {
	case "changes":
		if m, ok := v.(map[string]any); ok {
			return w.changes(m)
		}
	case "originSelectionRange":
		return w.rewrite(v, ctx)
	}
	return w.rewrite(v, own)
}

func (w *rewriter) changes(m map[string]any) map[string]any {
	out := make(map[string]any, len(m))
	for uri, v := range m {
		lw := w.route(uri)
		if lw != nil {
			uri = w.name(lw)
		}
		out[uri] = w.rewrite(v, lw)
	}
	return out
}

func number(v any) (int, bool) {
	n, ok := v.(json.Number)
	if !ok {
		return 0, false
	}
	i, err := n.Int64()
	return int(i), err == nil
}

func asPosition(m map[string]any) (Position, bool) {
	if len(m) != 2 {
		return Position{}, false
	}
	line, ok1 := number(m["line"])
	char, ok2 := number(m["character"])
	return Position{Line: line, Character: char}, ok1 && ok2
}

func asRange(m map[string]any) (Range, bool) {
	if len(m) != 2 {
		return Range{}, false
	}
	start, ok1 := m["start"].(map[string]any)
	end, ok2 := m["end"].(map[string]any)
	if !ok1 || !ok2 {
		return Range{}, false
	}
	s, ok3 := asPosition(start)
	e, ok4 := asPosition(end)
	return Range{Start: s, End: e}, ok3 && ok4
}

func positionJSON(p Position) map[string]any {
	return map[string]any{
		"line":      json.Number(strconv.Itoa(p.Line)),
		"character": json.Number(strconv.Itoa(p.Character)),
	}
}

func rangeJSON(r Range) map[string]any {
	return map[string]any{"start": positionJSON(r.Start), "end": positionJSON(r.End)}
}

func decode(raw json.RawMessage) (any, error) {
	if len(bytes.TrimSpace(raw)) == 0 {
		return nil, nil
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	var v any
	err := dec.Decode(&v)
	return v, err
}

func encode(v any) (json.RawMessage, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	return json.RawMessage(strings.TrimSuffix(buf.String(), "\n")), nil
}
