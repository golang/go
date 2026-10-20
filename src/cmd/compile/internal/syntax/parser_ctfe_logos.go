// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

// This file holds the loGo/s parser logic for M5-S2 (`const func`
// declarations; SPEC Sec 2.1, T-201). parser.go calls into it through a
// hook line only (G3); the rule below runs solely in LogosMode, so
// plain Go parses exactly as upstream. In plain mode `const` never
// precedes `func`, so the hook falls through to the upstream const
// group path and its missing-name error.

// logosConstDecl appends one file-scope `const` declaration. In
// LogosMode, `const func` declares a const function: the inner FuncDecl
// parses with the ordinary rules (so receivers, type parameters, and
// error recovery behave identically) and is wrapped with the position
// of `const` for the checker's purity gate and lowering's splice. The
// inner position keeps the `func` keyword invariant of FuncDecl. A
// pending pragma attaches to the inner function. Grouped `const`
// parentheses never hold a function; every other `const` keeps the
// upstream path.
func (p *parser) logosConstDecl(list []Decl) []Decl {
	pos := p.pos() // `const`
	p.next()
	if p.mode&LogosMode != 0 && p.tok == _Func {
		funcPos := p.pos() // `func`
		pragma := p.takePragma()
		p.next()
		fn := p.funcDeclOrNil()
		fn.pos = funcPos
		if fn.Pragma == nil {
			fn.Pragma = pragma
		}
		d := &ConstFuncDecl{ConstPos: pos, Func: fn}
		d.pos = pos
		return append(list, d)
	}
	return p.appendGroup(list, p.constDecl)
}
