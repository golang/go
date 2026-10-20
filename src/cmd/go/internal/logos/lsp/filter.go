// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import "strings"

// reservedPrefix starts every identifier the lowering invents (_logos_e0,
// _logos_rt, _logos_site0). The language reserves it, so no source of the
// user can name one, and gopls's answers about the lowered text must not
// show one to the user.
const reservedPrefix = "_logos_"

// insideGenerated reports whether the lowered offset o lies strictly inside
// text the lowering wrote. The first and last offsets of generated text are
// seams with the user's own text, and do not count.
func (m *posMap) insideGenerated(o int) bool {
	shift := 0
	for _, s := range m.spans {
		start := s.Start + shift
		if o <= start {
			return false
		}
		if o < start+s.Len {
			return true
		}
		shift += s.Len - (s.End - s.Start)
	}
	return false
}

// prepareResult removes from a decoded gopls result, still in the
// coordinates of the lowered text, what describes only text the lowering
// wrote: inlay hints placed inside generated text, such as the type of the
// error variable a ? statement declares, and completions and symbols named
// with the reserved prefix. A request on a Go document has no lowering and
// keeps its result.
func prepareResult(pend pending, result any) any {
	if pend.lw == nil {
		return result
	}
	switch pend.method {
	case "textDocument/inlayHint":
		return dropGeneratedHints(result, pend.lw)
	case "textDocument/completion":
		return dropReservedCompletions(result)
	case "textDocument/documentSymbol":
		return dropReservedSymbols(result)
	}
	return result
}

// dropGeneratedHints removes the inlay hints of a decoded answer whose
// position lies inside generated text of lw.
func dropGeneratedHints(result any, lw *lowered) any {
	list, ok := result.([]any)
	if !ok {
		return result
	}
	kept := make([]any, 0, len(list))
	for _, item := range list {
		if !hintInsideGenerated(item, lw) {
			kept = append(kept, item)
		}
	}
	return kept
}

// hintInsideGenerated reports whether the inlay hint item sits inside
// generated text of lw.
func hintInsideGenerated(item any, lw *lowered) bool {
	m, _ := item.(map[string]any)
	pos, _ := m["position"].(map[string]any)
	p, ok := asPosition(pos)
	return ok && lw.pm.insideGenerated(lw.pm.low.offset(p))
}

// dropReservedCompletions removes the completion items named with the
// reserved prefix from a decoded answer, a list or a completion list.
func dropReservedCompletions(result any) any {
	if list, ok := result.([]any); ok {
		return withoutReserved(list, "label")
	}
	if m, ok := result.(map[string]any); ok {
		if items, ok := m["items"].([]any); ok {
			m["items"] = withoutReserved(items, "label")
		}
	}
	return result
}

// dropReservedSymbols removes the document symbols named with the reserved
// prefix from a decoded answer, at every depth of a symbol tree.
func dropReservedSymbols(result any) any {
	list, ok := result.([]any)
	if !ok {
		return result
	}
	kept := withoutReserved(list, "name")
	for _, symbol := range kept {
		if m, ok := symbol.(map[string]any); ok {
			if children, ok := m["children"].([]any); ok {
				m["children"] = dropReservedSymbols(children)
			}
		}
	}
	return kept
}

// withoutReserved returns the items of list whose string member key does
// not start with the reserved prefix.
func withoutReserved(list []any, key string) []any {
	kept := make([]any, 0, len(list))
	for _, item := range list {
		m, _ := item.(map[string]any)
		if name, _ := m[key].(string); !strings.HasPrefix(name, reservedPrefix) {
			kept = append(kept, item)
		}
	}
	return kept
}
