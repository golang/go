// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

// This file holds the loGo/s parser logic for M6-S7 collection chains
// (SPEC Sec 1.4, T-104.6). parser.go calls into it through hook lines
// only (G3); the rule below runs solely in LogosMode, so plain Go
// parses exactly as upstream.

// logosDotMap parses the `map` in a `.map` chain call. `map` scans as
// a keyword, which can never be a plain selector, so no valid Go input
// parses differently; in plain mode it returns nil for the upstream
// "expected name or (" error.
func (p *parser) logosDotMap() *Name {
	if p.mode&LogosMode == 0 || p.tok != _Map {
		return nil
	}
	name := NewName(p.pos(), "map")
	p.next()
	return name
}
