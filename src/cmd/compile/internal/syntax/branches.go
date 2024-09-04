// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import "fmt"

// checkBranches checks correct use of labels and branch
// statements (break, continue, fallthrough, goto) in a function body.
// It catches:
//   - misplaced breaks, continues, and fallthroughs
//   - bad labeled breaks and continues
//   - invalid, unused, duplicate, and missing labels
//   - gotos jumping over variable declarations and into blocks
func checkBranches(body *BlockStmt, errh ErrorHandler) {
	if body == nil {
		return
	}

	// scope of all labels in this body
	ls := &labelScope{errh: errh}
	fwdGotos := ls.blockBranches(nil, targets{}, nil, body.Pos(), body.List)

	// If there are any forward gotos left, no matching label was
	// found for them. Either those labels were never defined, or
	// they are inside blocks and not reachable from the gotos.
	for _, fwd := range fwdGotos {
		name := fwd.Label.Value
		if l := ls.labels[name]; l != nil {
			l.used = true // avoid "defined and not used" error
			ls.errf(fwd.Label.Pos(), "goto %s jumps into block starting at %s", name, l.parent.start)
		} else {
			ls.errf(fwd.Label.Pos(), "label %s not defined", name)
		}
	}

	// spec: "It is illegal to define a label that is never used."
	for _, l := range ls.labels {
		if !l.used {
			l := l.lstmt.Label
			ls.errf(l.Pos(), "label %s defined and not used", l.Value)
		}
	}
}

type labelScope struct {
	errh   ErrorHandler
	labels map[string]*label // all label declarations inside the function; allocated lazily
}

type label struct {
	parent *block       // block containing this label declaration
	lstmt  *LabeledStmt // statement declaring the label
	used   bool         // whether the label is used or not
}

type block struct {
	parent *block       // immediately enclosing block, or nil
	start  Pos          // start of block
	lstmt  *LabeledStmt // labeled statement associated with this block, or nil
}

func (ls *labelScope) errf(pos Pos, format string, args ...interface{}) {
	ls.errh(Error{pos, fmt.Sprintf(format, args...)})
}

// declare declares the label introduced by s in block b and returns
// the new label. If the label was already declared, declare reports
// and error and the existing label is returned instead.
func (ls *labelScope) declare(b *block, s *LabeledStmt) *label {
	name := s.Label.Value
	labels := ls.labels
	if labels == nil {
		labels = make(map[string]*label)
		ls.labels = labels
	} else if alt := labels[name]; alt != nil {
		ls.errf(s.Label.Pos(), "label %s already defined at %s", name, alt.lstmt.Label.Pos().String())
		return alt
	}
	l := &label{b, s, false}
	labels[name] = l
	return l
}

// gotoTarget returns the labeled statement matching the given name and
// declared in block b or any of its enclosing blocks. The result is nil
// if the label is not defined, or doesn't match a valid labeled statement.
func (ls *labelScope) gotoTarget(b *block, name string) *LabeledStmt {
	if l := ls.labels[name]; l != nil {
		l.used = true // even if it's not a valid target
		for ; b != nil; b = b.parent {
			if l.parent == b {
				return l.lstmt
			}
		}
	}
	return nil
}

var invalid = new(LabeledStmt) // singleton to signal invalid enclosing target

// enclosingTarget returns the innermost enclosing labeled statement matching
// the given name. The result is nil if the label is not defined, and invalid
// if the label is defined but doesn't label a valid labeled statement.
func (ls *labelScope) enclosingTarget(b *block, name string) *LabeledStmt {
	if l := ls.labels[name]; l != nil {
		l.used = true // even if it's not a valid target (see e.g., test/fixedbugs/bug136.go)
		for ; b != nil; b = b.parent {
			if l.lstmt == b.lstmt {
				return l.lstmt
			}
		}
		return invalid
	}
	return nil
}

// targets describes the target statements within which break
// or continue statements are valid.
type targets struct {
	breaks    Stmt     // *ForStmt, *SwitchStmt, *SelectStmt, or nil
	continues *ForStmt // or nil
	caseIndex int      // case index of immediately enclosing switch statement, or < 0
}

// blockBranches processes a block's body starting at start and returns the
// list of unresolved (forward) gotos. parent is the immediately enclosing
// block (or nil), ctxt provides information about the enclosing statements,
// and lstmt is the labeled statement associated with this block, or nil.
func (ls *labelScope) blockBranches(parent *block, ctxt targets, lstmt *LabeledStmt, start Pos, body []Stmt) []*BranchStmt {
	b := &block{parent: parent, start: start, lstmt: lstmt}

	var varPos Pos
	var varName Expr
	var fwdGotos, badGotos []*BranchStmt

	recordVarDecl := func(pos Pos, name Expr) {
		varPos = pos
		varName = name
		// Any existing forward goto jumping over the variable
		// declaration is invalid. The goto may still jump out
		// of the block and be ok, but we don't know that yet.
		// Remember all forward gotos as potential bad gotos.
		badGotos = append(badGotos[:0], fwdGotos...)
	}

	jumpsOverVarDecl := func(fwd *BranchStmt) bool {
		if varPos.IsKnown() {
			for _, bad := range badGotos {
				if fwd == bad {
					return true
				}
			}
		}
		return false
	}

	innerBlock := func(ctxt targets, start Pos, body []Stmt) {
		// Unresolved forward gotos from the inner block
		// become forward gotos for the current block.
		fwdGotos = append(fwdGotos, ls.blockBranches(b, ctxt, lstmt, start, body)...)
	}

	// A fallthrough statement counts as last statement in a statement
	// list even if there are trailing empty statements; remove them.
	stmtList := trimTrailingEmptyStmts(body)
	for stmtIndex, stmt := range stmtList {
		lstmt = nil
	L:
		switch s := stmt.(type) {
		case *DeclStmt:
			for _, d := range s.DeclList {
				if v, ok := d.(*VarDecl); ok {
					recordVarDecl(v.Pos(), v.NameList[0])
					break // the first VarDecl will do
				}
			}

		case *LabeledStmt:
			// declare non-blank label
			if name := s.Label.Value; name != "_" {
				l := ls.declare(b, s)
				// resolve matching forward gotos
				i := 0
				for _, fwd := range fwdGotos {
					if fwd.Label.Value == name {
						fwd.Target = s
						l.used = true
						if jumpsOverVarDecl(fwd) {
							ls.errf(
								fwd.Label.Pos(),
								"goto %s jumps over declaration of %s at %s",
								name, String(varName), varPos,
							)
						}
					} else {
						// no match - keep forward goto
						fwdGotos[i] = fwd
						i++
					}
				}
				fwdGotos = fwdGotos[:i]
				lstmt = s
			}
			// process labeled statement
			stmt = s.Stmt
			goto L

		case *BranchStmt:
			// unlabeled branch statement
			if s.Label == nil {
				switch s.Tok {
				case _Break:
					if t := ctxt.breaks; t != nil {
						s.Target = t
					} else {
						ls.errf(s.Pos(), "break is not in a loop, switch, or select")
					}
				case _Continue:
					if t := ctxt.continues; t != nil {
						s.Target = t
					} else {
						ls.errf(s.Pos(), "continue is not in a loop")
					}
				case _Fallthrough:
					msg := "fallthrough statement out of place"
					if t, _ := ctxt.breaks.(*SwitchStmt); t != nil {
						if _, ok := t.Tag.(*TypeSwitchGuard); ok {
							msg = "cannot fallthrough in type switch"
						} else if ctxt.caseIndex < 0 || stmtIndex+1 < len(stmtList) {
							// fallthrough nested in a block or not the last statement
							// use msg as is
						} else if ctxt.caseIndex+1 == len(t.Body) {
							msg = "cannot fallthrough final case in switch"
						} else {
							break // fallthrough ok
						}
					}
					ls.errf(s.Pos(), "%s", msg)
				case _Goto:
					fallthrough // should always have a label
				default:
					panic("invalid BranchStmt")
				}
				break
			}

			// labeled branch statement
			name := s.Label.Value
			switch s.Tok {
			case _Break:
				// spec: "If there is a label, it must be that of an enclosing
				// "for", "switch", or "select" statement, and that is the one
				// whose execution terminates."
				if t := ls.enclosingTarget(b, name); t != nil {
					switch t := t.Stmt.(type) {
					case *SwitchStmt, *SelectStmt, *ForStmt:
						s.Target = t
					default:
						ls.errf(s.Label.Pos(), "invalid break label %s", name)
					}
				} else {
					ls.errf(s.Label.Pos(), "break label not defined: %s", name)
				}

			case _Continue:
				// spec: "If there is a label, it must be that of an enclosing
				// "for" statement, and that is the one whose execution advances."
				if t := ls.enclosingTarget(b, name); t != nil {
					if t, ok := t.Stmt.(*ForStmt); ok {
						s.Target = t
					} else {
						ls.errf(s.Label.Pos(), "invalid continue label %s", name)
					}
				} else {
					ls.errf(s.Label.Pos(), "continue label not defined: %s", name)
				}

			case _Goto:
				if t := ls.gotoTarget(b, name); t != nil {
					s.Target = t
				} else {
					// label may be declared later - add goto to forward gotos
					fwdGotos = append(fwdGotos, s)
				}

			case _Fallthrough:
				fallthrough // should never have a label
			default:
				panic("invalid BranchStmt")
			}

		case *AssignStmt:
			if s.Op == Def {
				recordVarDecl(s.Pos(), s.Lhs)
			}

		case *BlockStmt:
			inner := targets{ctxt.breaks, ctxt.continues, -1}
			innerBlock(inner, s.Pos(), s.List)

		case *IfStmt:
			inner := targets{ctxt.breaks, ctxt.continues, -1}
			innerBlock(inner, s.Then.Pos(), s.Then.List)
			if s.Else != nil {
				innerBlock(inner, s.Else.Pos(), []Stmt{s.Else})
			}

		case *ForStmt:
			inner := targets{s, s, -1}
			innerBlock(inner, s.Body.Pos(), s.Body.List)

		case *SwitchStmt:
			inner := targets{s, ctxt.continues, -1}
			for i, cc := range s.Body {
				inner.caseIndex = i
				innerBlock(inner, cc.Pos(), cc.Body)
			}

		case *SelectStmt:
			inner := targets{s, ctxt.continues, -1}
			for _, cc := range s.Body {
				innerBlock(inner, cc.Pos(), cc.Body)
			}
		}
	}

	return fwdGotos
}

func trimTrailingEmptyStmts(list []Stmt) []Stmt {
	for i := len(list); i > 0; i-- {
		if _, ok := list[i-1].(*EmptyStmt); !ok {
			return list[:i]
		}
	}
	return nil
}
