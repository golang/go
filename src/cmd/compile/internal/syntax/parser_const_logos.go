// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

// This file holds the loGo/s parser logic for M5-S1 (const views
// `const T`; SPEC Sec 2.1, T-201). parser.go calls into it through a
// hook line only (G3); the rule below runs solely in LogosMode, so
// plain Go parses exactly as upstream. In plain mode `const` never
// starts a type, so the hook returns nil and the caller's missing-type
// error path runs unchanged.

// logosConstType parses a `const T` const-view type. The caller saw
// `const` where a type starts (parameter, receiver, result, variable,
// or field type) in LogosMode; T itself parses as a full type, so
// `const []int?` views an option type. In plain mode it returns nil.
func (p *parser) logosConstType() Expr {
	if p.mode&LogosMode == 0 {
		return nil
	}
	pos := p.pos()
	p.next() // const
	t := new(ConstType)
	t.pos = pos
	t.Elem = p.type_()
	return t
}
