// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import (
	"cmd/internal/src"
	"fmt"
)

// TODO(gri) do this while parsing instead of in a separate pass?

// checkBranches checks correct use of labels and branch
// statements (break, continue, goto) in a function body.
// It catches:
//    - misplaced breaks and continues
//    - bad labeled breaks and continues
//    - invalid, unused, duplicate, and missing labels
//    - gotos jumping over variable declarations and into blocks
func checkBranches(body *BlockStmt, errh ErrorHandler) {
	if body == nil {
		return
	}

	// scope of all labels in this body
	ls := &labelScope{errh: errh}
	fwdGo2s := ls.blockBranches(nil, 0, nil, body.Pos(), body.List)

	// If there are any forward gotos left, no matching label was
	// found for them. Either those labels were never defined, or
	// they are inside blocks and not reachable from the gotos.
	for _, go2 := range fwdGo2s {
		name := go2.Label.Value
		if l := ls.labels[name]; l != nil {
			l.used = true // avoid "defined and not used" error
			ls.err(go2.Label.Pos(), "goto %s jumps into block starting at %s", name, l.parent.start)
		} else {
			ls.err(go2.Label.Pos(), "label %s not defined", name)
		}
	}

	// spec: "It is illegal to define a label that is never used."
	for _, l := range ls.labels {
		if !l.used {
			l := l.lstmt.Label
			ls.err(l.Pos(), "label %s defined and not used", l.Value)
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
	start  src.Pos      // start of block
	lstmt  *LabeledStmt // labeled statement associated with this block, or nil
}

func (ls *labelScope) err(pos src.Pos, format string, args ...interface{}) {
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
		ls.err(s.Pos(), "label %s already defined at %s", name, alt.lstmt.Label.Pos().String())
		return alt
	}
	l := &label{b, s, false}
	labels[name] = l
	return l
}

// gotoTarget returns the labeled statement matching the given name and
// declared in block b or any of its enclosing blocks. The result is nil
// if the label is not defined, or doesn't match a valid labeled statement.
func (ls *labelScope) gotoTarget(b *block, name string) *label {
	if l := ls.labels[name]; l != nil {
		l.used = true // even if it's not a valid target
		for ; b != nil; b = b.parent {
			if l.parent == b {
				return l
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

// context flags
const (
	breakOk = 1 << iota
	continueOk
)

// blockBranches processes a block's body starting at start and returns the
// list of unresolved (forward) gotos. parent is the immediately enclosing
// block (or nil), context provides information about the enclosing statements,
// and lstmt is the labeled statement asociated with this block, or nil.
func (ls *labelScope) blockBranches(parent *block, context uint, lstmt *LabeledStmt, start src.Pos, body []Stmt) []*BranchStmt {
	b := &block{parent: parent, start: start, lstmt: lstmt}

	var varPos src.Pos
	var varName Expr
	var fwdGo2s, badGo2s []*BranchStmt

	recordVarDecl := func(pos src.Pos, name Expr) {
		varPos = pos
		varName = name
		// Any existing forward goto jumping over the variable
		// declaration is invalid. The goto may still jump out
		// of the block and be ok, but we don't know that yet.
		// Remember all forward gotos as potential bad gotos.
		badGo2s = append(badGo2s[:0], fwdGo2s...)
	}

	jumpsOverVarDecl := func(go2 *BranchStmt) bool {
		if varPos.IsKnown() {
			for _, bad := range badGo2s {
				if go2 == bad {
					return true
				}
			}
		}
		return false
	}

	innerBlock := func(flags uint, start src.Pos, body []Stmt) {
		fwdGo2s = append(fwdGo2s, ls.blockBranches(b, context|flags, lstmt, start, body)...)
	}

	for _, stmt := range body {
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
				for _, go2 := range fwdGo2s {
					if go2.Label.Value == name {
						l.used = true
						if jumpsOverVarDecl(go2) {
							ls.err(
								go2.Label.Pos(),
								"goto %s jumps over declaration of %s at %s",
								name, String(varName), varPos,
							)
						}
					} else {
						// no match - keep forward goto
						fwdGo2s[i] = go2
						i++
					}
				}
				fwdGo2s = fwdGo2s[:i]
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
					if context&breakOk == 0 {
						ls.err(s.Pos(), "break is not in a loop, switch, or select")
					}
				case _Continue:
					if context&continueOk == 0 {
						ls.err(s.Pos(), "continue is not in a loop")
					}
				case _Fallthrough:
					// nothing to do
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
					valid := false
					switch t.Stmt.(type) {
					case *SwitchStmt, *SelectStmt, *ForStmt:
						valid = true
					}
					if !valid {
						ls.err(s.Label.Pos(), "invalid break label %s", name)
					}
				} else {
					ls.err(s.Label.Pos(), "break label not defined: %s", name)
				}

			case _Continue:
				// spec: "If there is a label, it must be that of an enclosing
				// "for" statement, and that is the one whose execution advances."
				if t := ls.enclosingTarget(b, name); t != nil {
					if _, ok := t.Stmt.(*ForStmt); !ok {
						ls.err(s.Label.Pos(), "invalid continue label %s", name)
					}
				} else {
					ls.err(s.Label.Pos(), "continue label not defined: %s", name)
				}

			case _Goto:
				if ls.gotoTarget(b, name) == nil {
					// label may be declared later - add goto to forward gotos
					fwdGo2s = append(fwdGo2s, s)
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
			// Unresolved forward gotos from the nested block
			// become forward gotos for the current block.
			innerBlock(0, s.Pos(), s.List)

		case *IfStmt:
			innerBlock(0, s.Then.Pos(), s.Then.List)
			if s.Else != nil {
				innerBlock(0, s.Else.Pos(), []Stmt{s.Else})
			}

		case *ForStmt:
			innerBlock(breakOk|continueOk, s.Body.Pos(), s.Body.List)

		case *SwitchStmt:
			for _, cc := range s.Body {
				innerBlock(breakOk, cc.Pos(), cc.Body)
			}

		case *SelectStmt:
			for _, cc := range s.Body {
				innerBlock(breakOk, cc.Pos(), cc.Body)
			}
		}
	}

	return fwdGo2s
}
