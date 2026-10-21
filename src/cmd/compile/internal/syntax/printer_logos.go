// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

// This file holds the loGo/s printing logic for M1-S2 and M2 (G3: hook
// lines only in printer.go; the logic lives here).

// printLogos prints loGo/s nodes in source form: `..src` for SpreadExpr,
// `X?` for TryExpr,
// `T?` for OptionType, `x ?? y` for CoalesceExpr, `x?.Sel` for
// OptSelectExpr, `const T` for ConstType, `errdefer <call>` for
// ErrdeferStmt, `switch ... { case ... => ... }` for SwitchExpr and
// ArmClause, and `struct : I, J { ... }` for AssertedStruct. Every form
// reparses to the same tree, so printing is idempotent. ShortForm
// elides the arms of a SwitchExpr like the body of a function literal.
func printLogos(p *printer, n Node) {
	switch n := n.(type) {
	case *TryExpr:
		p.print(n.X, _Question)
	case *SpreadExpr:
		p.print(_Dot, _Dot, n.X)
	case *InterpExpr:
		// The raw spelling reparse to the same tree, so printing
		// stays idempotent (like BasicLit, which prints its Value).
		p.print(_Name, n.Raw)
	case *OptionType:
		// A nested option needs parentheses (`(T?)?`), because the
		// joined spelling `T??` is a syntax error (SPEC T-202).
		if _, ok := n.Elem.(*OptionType); ok {
			p.print(_Lparen, n.Elem, _Rparen, _Question)
		} else {
			p.print(n.Elem, _Question)
		}
	case *CoalesceExpr:
		// Flat like a binary Operation; precedence parens stay the
		// caller's house-wide TODO.
		p.print(n.X, blank, _Coalesce, blank, n.Y)
	case *OptSelectExpr:
		p.print(n.X, _QuestionDot, n.Sel)
	case *ConstType:
		p.print(_Const, blank, n.Elem)
	case *ConstFuncDecl: // M5-S2: `const func` reparses through the parser hook.
		p.print(_Const, blank, n.Func)
	case *ErrdeferStmt:
		p.print(_Name, "errdefer", blank, n.Call)
	case *YieldStmt:
		p.print(_Name, "yield", blank, n.Value)
	case *ScopeDeferStmt:
		p.print(_Name, "scope", blank, _Defer, blank, n.Call)
	case *GroupStmt: // M8-S2: `go group (ctx)` prints its source form.
		p.print(_Go, blank, _Name, "group", blank, _Lparen, n.Ctx)
		if n.Limit != nil {
			p.print(_Comma, blank, _Name, "limit", blank, _Assign, blank, n.Limit)
		}
		p.print(_Rparen, blank, n.Body)
	case *GroupChild: // M8-S2: `go { }` prints its source form.
		p.print(_Go, blank, n.Body)
	case *SwitchExpr:
		p.print(_Switch, blank)
		if n.Init != nil {
			p.print(n.Init, _Semi, blank)
		}
		if n.Tag != nil {
			p.print(n.Tag, blank)
		}
		p.print(_Lbrace)
		switch {
		case p.form == ShortForm:
			if len(n.Arms) > 0 {
				p.print(_Name, "…")
			}
		case len(n.Arms) > 0:
			p.print(newline)
			for _, arm := range n.Arms {
				p.print(arm, _Semi, newline)
			}
		}
		p.print(_Rbrace)
	case *ArmClause:
		if n.Cases != nil {
			p.print(_Case, blank, n.Cases)
		} else {
			p.print(_Default)
		}
		p.print(blank, _FatArrow, blank, n.Body)
	case *PropertyType:
		if n.Const {
			p.print(_Const, blank)
		}
		p.printNode(n.Type)
	case *FieldDefault:
		p.print(n.Type, blank, _Assign, blank, n.Default)
	case *TupleType:
		p.print(_Lparen)
		for i, f := range n.Fields {
			if i > 0 {
				p.print(_Comma, blank)
			}
			if f.Name != nil {
				p.printNode(f.Name)
				p.print(blank)
			}
			p.printNode(f.Type)
		}
		p.print(_Rparen)
	case *TupleLit:
		p.print(_Lparen)
		p.printExprList(n.Elems)
		p.print(_Rparen)
	case *LambdaExpr:
		if len(n.Params) == 1 {
			p.printNode(n.Params[0])
		} else {
			p.print(_Lparen)
			for i, name := range n.Params {
				if i > 0 {
					p.print(_Comma, blank)
				}
				p.printNode(name)
			}
			p.print(_Rparen)
		}
		p.print(blank, _FatArrow, blank)
		p.printNode(n.Body)
	case *ConstIfStmt:
		p.print(_Const, blank, _If, blank, n.Cond, blank, n.Then)
		if n.Else != nil {
			p.print(blank, _Else, blank, n.Else)
		}
	case *ConstSwitchStmt:
		p.print(_Const, blank, _Switch, blank)
		p.printNode(n.Targ)
		p.print(blank, _Lbrace)
		if len(n.Cases) > 0 {
			p.print(newline)
			for i, c := range n.Cases {
				p.print(_Case, blank)
				if c.Tilde {
					p.print(Tilde)
				}
				p.printNode(c.Type)
				p.print(_Colon)
				if len(c.Body) > 0 {
					p.print(newline, indent)
					p.printStmtList(c.Body, i+1 == len(n.Cases))
					p.print(outdent)
				}
				p.print(newline)
			}
		}
		p.print(_Rbrace)
	case *ConstForStmt:
		p.print(_Const, blank, _For, blank)
		if n.Index != nil {
			p.printNode(n.Index)
			p.print(_Comma, blank)
		}
		p.printNode(n.Field)
		p.print(blank, _Define, blank, _Range, blank)
		p.printNode(n.Range)
		p.print(blank)
		p.printNode(n.Body)
	case *FieldProjExpr:
		p.printNode(n.X)
		p.print(_Dot, _Lbrack)
		p.printNode(n.Field)
		p.print(_Rbrack)
	case *AssertedStruct:
		p.print(_Struct, blank, _Colon, blank)
		p.printExprList(n.Asserts)
		p.print(blank, _Lbrace)
		if fields := n.Struct.FieldList; len(fields) > 0 {
			if p.linebreaks {
				p.print(newline, indent)
				p.printFieldList(fields, n.Struct.TagList, _Semi)
				p.print(outdent, newline)
			} else {
				p.printFieldList(fields, n.Struct.TagList, _Semi)
			}
		}
		p.print(_Rbrace)
	}
}

// logosPrintProperty prints one property interface element in source
// form: `const F T` for a read-only property, `F T` otherwise. Both
// reparse to the same tree, so printing stays idempotent.
func logosPrintProperty(p *printer, name *Name, pt *PropertyType) {
	if pt.Const {
		p.print(_Const, blank)
	}
	p.printNode(name)
	p.print(blank)
	p.printNode(pt.Type)
}

// logosImpliesSemi reports whether a loGo/s token ends an operand like a
// name or `)`: `?` implies a semicolon, so the printer elides the
// redundant `;` after `?` between statements.
func logosImpliesSemi(tok token) bool {
	return tok == _Question
}
