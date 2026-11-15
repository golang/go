// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"strings"

	"cmd/internal/logos/editmap"
)

const reservedPrefix = "_logos_"

func (m *posMap) insideGenerated(o int) bool {
	_, generated := m.OriginalOffset(o, editmap.Leading)
	return generated
}

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

func dropGeneratedHints(result any, lw *lowered) any {
	list, ok := result.([]any)
	if !ok {
		return result
	}
	kept := make([]any, 0, len(list))
	for _, item := range list {
		m, _ := item.(map[string]any)
		pos, _ := m["position"].(map[string]any)
		p, ok := asPosition(pos)
		if !ok || !lw.pm.insideGenerated(lw.pm.low.offset(p)) {
			kept = append(kept, item)
		}
	}
	return kept
}

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
