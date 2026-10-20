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

// lowered is the lowering of one loGo/s file: the position map between its
// text and the Go document gopls sees for it. A lowering is fresh when it
// was made from the text the client holds; when the file has changed since,
// or a later compile of it failed, the old lowering stays for the packages
// that depend on it but the proxy stops answering position requests from
// it.
type lowered struct {
	src   string
	virt  string
	pm    *posMap
	fresh bool
}

// srcURI returns the URI of the loGo/s source.
func (lw *lowered) srcURI() string { return pathURI(lw.src) }

// virtURI returns the URI of the lowered Go document.
func (lw *lowered) virtURI() string { return pathURI(lw.virt) }

// tables are the lowerings of the open packages, rebuilt on every
// recompile and read without a lock.
type tables struct {
	bySrc  map[string]*lowered
	byVirt map[string]*lowered
}

// toLowered returns the rewriter for messages from the client to gopls.
func (tb *tables) toLowered() *rewriter {
	return &rewriter{
		route: func(uri string) *lowered { return tb.lookup(tb.bySrc, uri) },
		name:  (*lowered).virtURI,
		pos:   func(lw *lowered, p Position) Position { return lw.pm.toLowered(p, leading) },
		rng:   func(lw *lowered, r Range) Range { return lw.pm.rangeToLowered(r) },
	}
}

// toSource returns the rewriter for messages from gopls to the client.
// Versions of lowered documents mean nothing to the client, so it drops
// them.
func (tb *tables) toSource() *rewriter {
	return &rewriter{
		route:       func(uri string) *lowered { return tb.lookup(tb.byVirt, uri) },
		name:        (*lowered).srcURI,
		pos:         func(lw *lowered, p Position) Position { return lw.pm.toOriginal(p, leading) },
		rng:         func(lw *lowered, r Range) Range { return lw.pm.rangeToOriginal(r) },
		dropVersion: true,
	}
}

// lookup returns the lowering that the file URI uri names in m.
func (tb *tables) lookup(m map[string]*lowered, uri string) *lowered {
	path, err := uriPath(uri)
	if err != nil {
		return nil
	}
	return m[path]
}

// rewriter translates a JSON value between the two coordinate spaces of
// the proxy: the loGo/s sources the client sees and the lowered documents
// gopls sees. It walks the value, renames every URI that names a lowered
// file, and maps every range and position through the position map of the
// file it belongs to. A range belongs to the nearest enclosing object that
// names a file (a Location, a LocationLink target, a text document edit);
// outside one it belongs to the file of the request, and a range of a file
// with no lowering passes unchanged.
type rewriter struct {
	route       func(uri string) *lowered
	name        func(lw *lowered) string
	pos         func(lw *lowered, p Position) Position
	rng         func(lw *lowered, r Range) Range
	dropVersion bool
}

// rewrite translates v, which belongs to the file of ctx (nil for none).
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

// object translates one JSON object.
func (w *rewriter) object(m map[string]any, ctx *lowered) any {
	if r, ok := asRange(m); ok {
		return w.rangeValue(m, r, ctx)
	}
	if p, ok := asPosition(m); ok {
		return w.positionValue(m, p, ctx)
	}
	own := w.owner(m, ctx)
	w.renameURIs(m)
	for key, v := range m {
		m[key] = w.child(key, v, own, ctx)
	}
	return m
}

// rangeValue maps the range r held in m, or leaves m when there is no file.
func (w *rewriter) rangeValue(m map[string]any, r Range, ctx *lowered) any {
	if ctx == nil {
		return m
	}
	return rangeJSON(w.rng(ctx, r))
}

// positionValue maps the position p held in m, or leaves m when there is
// no file.
func (w *rewriter) positionValue(m map[string]any, p Position, ctx *lowered) any {
	if ctx == nil {
		return m
	}
	return positionJSON(w.pos(ctx, p))
}

// owner returns the file the children of m belong to: the file m names, or
// ctx when it names none.
func (w *rewriter) owner(m map[string]any, ctx *lowered) *lowered {
	if uri, ok := namedURI(m); ok {
		return w.route(uri)
	}
	return ctx
}

// namedURI returns the URI of the file m names, directly or through its
// textDocument member.
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

// renameURIs renames the URIs m holds that name a lowered file.
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

// child translates the member key of an object. own is the file of the
// object's members, ctx the file the object itself belongs to.
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

// changes translates the changes member of a workspace edit, an object
// keyed by file URI.
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

// number returns the integer value of a decoded JSON number.
func number(v any) (int, bool) {
	n, ok := v.(json.Number)
	if !ok {
		return 0, false
	}
	i, err := n.Int64()
	return int(i), err == nil
}

// asPosition recognizes an LSP position object.
func asPosition(m map[string]any) (Position, bool) {
	if len(m) != 2 {
		return Position{}, false
	}
	line, ok1 := number(m["line"])
	char, ok2 := number(m["character"])
	return Position{Line: line, Character: char}, ok1 && ok2
}

// asRange recognizes an LSP range object.
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

// positionJSON returns the decoded JSON form of p.
func positionJSON(p Position) map[string]any {
	return map[string]any{
		"line":      json.Number(strconv.Itoa(p.Line)),
		"character": json.Number(strconv.Itoa(p.Character)),
	}
}

// rangeJSON returns the decoded JSON form of r.
func rangeJSON(r Range) map[string]any {
	return map[string]any{"start": positionJSON(r.Start), "end": positionJSON(r.End)}
}

// decode parses raw, keeping numbers exact. An empty raw decodes to nil.
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

// encode renders v as JSON without HTML escaping.
func encode(v any) (json.RawMessage, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	return json.RawMessage(strings.TrimSuffix(buf.String(), "\n")), nil
}
