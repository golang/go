// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hints

import (
	"cmp"
	"slices"
	"strings"
)

// InlayHints returns the inlay hints of the file named file that lie in rng:
// the narrowed type of each narrowed use, the member list at each union type
// switch, and the allocation tier of each allocation (SPEC Sec 21.2). A file
// the report does not cover has none. The hints come in position order.
func InlayHints(r *Report, file string, rng Range) []InlayHint {
	f := r.file(file)
	if f == nil {
		return nil
	}
	var out []InlayHint
	for _, n := range f.Narrowed {
		out = appendIn(out, rng, narrowedHint(n))
	}
	for _, u := range f.Unions {
		out = appendIn(out, rng, unionHint(u))
	}
	for _, a := range f.Allocs {
		out = appendIn(out, rng, allocHint(a))
	}
	slices.SortStableFunc(out, func(a, b InlayHint) int {
		return cmp.Or(cmp.Compare(a.Position.Line, b.Position.Line), cmp.Compare(a.Position.Character, b.Position.Character))
	})
	return out
}

// appendIn appends h to hs when its position lies in rng.
func appendIn(hs []InlayHint, rng Range, h InlayHint) []InlayHint {
	if rng.touches(h.Position) {
		return append(hs, h)
	}
	return hs
}

// narrowedHint renders `/* narrowed: *User */` after a narrowed use.
func narrowedHint(n Narrowed) InlayHint {
	return InlayHint{
		Position:    n.Position,
		Label:       "/* narrowed: " + n.Type + " */",
		Kind:        KindType,
		Tooltip:     n.Name + " is known to be non-nil here",
		PaddingLeft: true,
	}
}

// unionHint renders `/* union: A | B */` at a union type switch.
func unionHint(u Union) InlayHint {
	return InlayHint{
		Position:    u.Position,
		Label:       "/* union: " + strings.Join(u.Members, " | ") + " */",
		Kind:        KindType,
		Tooltip:     "closed union " + u.Type,
		PaddingLeft: true,
	}
}

// allocHint renders the tier badge `[heap]`, `[stack]`, `[arena]`, or
// `[rodata]`; the tooltip is the compiler's -m text and its reason.
func allocHint(a Alloc) InlayHint {
	tip := a.Message
	if a.Reason != "" {
		tip += " (" + a.Reason + ")"
	}
	return InlayHint{
		Position:     a.Position,
		Label:        "[" + a.Tier + "]",
		Tooltip:      tip,
		PaddingRight: true,
	}
}
