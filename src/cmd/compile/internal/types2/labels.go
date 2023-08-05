// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	. "internal/types/errors"
)

// labels checks correct label use in body.
func (check *Checker) labels(body *syntax.BlockStmt) {
	// set of all labels in this body
	all := NewScope(nil, body.Pos(), syntax.EndPos(body), "label")

	fwdJumps := check.blockBranches(all, nil, nil, body.List)

	// If there are any forward jumps left, no label was found for
	// the corresponding goto statements. Either those labels were
	// never defined, or they are inside blocks and not reachable
	// for the respective gotos.
	for _, jmp := range fwdJumps {
		var msg string
		var code Code
		name := jmp.Label.Value
		if alt := all.Lookup(name); alt != nil {
			msg = "goto %s jumps into block"
			alt.(*Label).used = true // avoid another error
			code = JumpIntoBlock
		} else {
			msg = "label %s not declared"
			code = UndeclaredLabel
		}
		check.errorf(jmp.Label, code, msg, name)
	}

	// spec: "It is illegal to define a label that is never used."
	for name, obj := range all.elems {
		obj = resolve(name, obj)
		if lbl := obj.(*Label); !lbl.used {
			check.softErrorf(lbl.pos, UnusedLabel, "label %s declared and not used", lbl.name)
		}
	}
}

// A block tracks label declarations in a block and its enclosing blocks.
type block struct {
	parent *block                         // enclosing block
	lstmt  *syntax.LabeledStmt            // labeled statement to which this block belongs, or nil
	labels map[string]*syntax.LabeledStmt // allocated lazily
}

// insert records a new label declaration for the current block.
// The label must not have been declared before in any block.
func (b *block) insert(s *syntax.LabeledStmt) {
	name := s.Label.Value
	if debug {
		assert(b.gotoTarget(name) == nil)
	}
	labels := b.labels
	if labels == nil {
		labels = make(map[string]*syntax.LabeledStmt)
		b.labels = labels
	}
	labels[name] = s
}

// gotoTarget returns the labeled statement in the current
// or an enclosing block with the given label name, or nil.
func (b *block) gotoTarget(name string) *syntax.LabeledStmt {
	for s := b; s != nil; s = s.parent {
		if t := s.labels[name]; t != nil {
			return t
		}
	}
	return nil
}

// enclosingTarget returns the innermost enclosing labeled
// statement with the given label name, or nil.
func (b *block) enclosingTarget(name string) *syntax.LabeledStmt {
	for s := b; s != nil; s = s.parent {
		if t := s.lstmt; t != nil && t.Label.Value == name {
			return t
		}
	}
	return nil
}

// blockBranches processes a block's statement list and returns the set of outgoing forward jumps.
// all is the scope of all declared labels, parent the set of labels declared in the immediately
// enclosing block, and lstmt is the labeled statement this block is associated with (or nil).
func (check *Checker) blockBranches(all *Scope, parent *block, lstmt *syntax.LabeledStmt, list []syntax.Stmt) []*syntax.BranchStmt {
	b := &block{parent, lstmt, nil}

	var (
		varDeclPos         syntax.Pos
		fwdJumps, badJumps []*syntax.BranchStmt
	)

	// All forward jumps jumping over a variable declaration are possibly
	// invalid (they may still jump out of the block and be ok).
	// recordVarDecl records them for the given position.
	recordVarDecl := func(pos syntax.Pos) {
		varDeclPos = pos
		badJumps = append(badJumps[:0], fwdJumps...) // copy fwdJumps to badJumps
	}

	jumpsOverVarDecl := func(jmp *syntax.BranchStmt) bool {
		if varDeclPos.IsKnown() {
			for _, bad := range badJumps {
				if jmp == bad {
					return true
				}
			}
		}
		return false
	}

	var stmtBranches func(syntax.Stmt)
	stmtBranches = func(s syntax.Stmt) {
		switch s := s.(type) {
		case *syntax.DeclStmt:
			for _, d := range s.DeclList {
				if d, _ := d.(*syntax.VarDecl); d != nil {
					recordVarDecl(d.Pos())
				}
			}

		case *syntax.LabeledStmt:
			// declare non-blank label
			if name := s.Label.Value; name != "_" {
				lbl := NewLabel(s.Label.Pos(), check.pkg, name)
				if alt := all.Insert(lbl); alt != nil {
					var err error_
					err.code = DuplicateLabel
					err.soft = true
					err.errorf(lbl.pos, "label %s already declared", name)
					err.recordAltDecl(alt)
					check.report(&err)
					// ok to continue
				} else {
					b.insert(s)
					check.recordDef(s.Label, lbl)
				}
				// resolve matching forward jumps and remove them from fwdJumps
				i := 0
				for _, jmp := range fwdJumps {
					if jmp.Label.Value == name {
						// match
						lbl.used = true
						check.recordUse(jmp.Label, lbl)
						if jumpsOverVarDecl(jmp) {
							check.softErrorf(
								jmp.Label,
								JumpOverDecl,
								"goto %s jumps over variable declaration at line %d",
								name,
								varDeclPos.Line(),
							)
							// ok to continue
						}
					} else {
						// no match - record new forward jump
						fwdJumps[i] = jmp
						i++
					}
				}
				fwdJumps = fwdJumps[:i]
				lstmt = s
			}
			stmtBranches(s.Stmt)

		case *syntax.BranchStmt:
			if s.Label == nil {
				return // checked in 1st pass (check.stmt)
			}

			// determine and validate target
			name := s.Label.Value
			switch s.Tok {
			case syntax.Break:
				// spec: "If there is a label, it must be that of an enclosing
				// "for", "switch", or "select" statement, and that is the one
				// whose execution terminates."
				valid := false
				if t := b.enclosingTarget(name); t != nil {
					switch t.Stmt.(type) {
					case *syntax.SwitchStmt, *syntax.SelectStmt, *syntax.ForStmt:
						valid = true
					}
				}
				if !valid {
					check.errorf(s.Label, MisplacedLabel, "invalid break label %s", name)
					return
				}

			case syntax.Continue:
				// spec: "If there is a label, it must be that of an enclosing
				// "for" statement, and that is the one whose execution advances."
				valid := false
				if t := b.enclosingTarget(name); t != nil {
					switch t.Stmt.(type) {
					case *syntax.ForStmt:
						valid = true
					}
				}
				if !valid {
					check.errorf(s.Label, MisplacedLabel, "invalid continue label %s", name)
					return
				}

			case syntax.Goto:
				if b.gotoTarget(name) == nil {
					// label may be declared later - add branch to forward jumps
					fwdJumps = append(fwdJumps, s)
					return
				}

			default:
				check.errorf(s, InvalidSyntaxTree, "branch statement: %s %s", s.Tok, name)
				return
			}

			// record label use
			obj := all.Lookup(name)
			obj.(*Label).used = true
			check.recordUse(s.Label, obj)

		case *syntax.AssignStmt:
			if s.Op == syntax.Def {
				recordVarDecl(s.Pos())
			}

		case *syntax.BlockStmt:
			// Unresolved forward jumps inside the nested block
			// become forward jumps in the current block.
			fwdJumps = append(fwdJumps, check.blockBranches(all, b, lstmt, s.List)...)

		case *syntax.IfStmt:
			stmtBranches(s.Then)
			if s.Else != nil {
				stmtBranches(s.Else)
			}

		case *syntax.SwitchStmt:
			b := &block{b, lstmt, nil}
			for _, s := range s.Body {
				fwdJumps = append(fwdJumps, check.blockBranches(all, b, nil, s.Body)...)
			}

		case *syntax.SelectStmt:
			b := &block{b, lstmt, nil}
			for _, s := range s.Body {
				fwdJumps = append(fwdJumps, check.blockBranches(all, b, nil, s.Body)...)
			}

		case *syntax.ForStmt:
			stmtBranches(s.Body)
		}
	}

	for _, s := range list {
		stmtBranches(s)
	}

	return fwdJumps
}
