// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"go/ast"
	"go/token"
	. "internal/types/errors"
)

// labels checks correct label use in body.
func (check *Checker) labels(body *ast.BlockStmt) {
	// set of all labels in this body
	all := NewScope(nil, body.Pos(), body.End(), "label")

	fwdJumps := check.blockBranches(all, nil, nil, body.List)

	// If there are any forward jumps left, no label was found for
	// the corresponding goto statements. Either those labels were
	// never defined, or they are inside blocks and not reachable
	// for the respective gotos.
	for _, jmp := range fwdJumps {
		var msg string
		var code Code
		name := jmp.Label.Name
		if alt := all.Lookup(name); alt != nil {
			msg = "goto %s jumps into block"
			alt.(*Label).used = true // avoid another error
			code = JumpIntoBlock
			// don't quote name here because "goto L" matches the code
		} else {
			msg = "label %s not declared"
			code = UndeclaredLabel
			name = quote(name)
		}
		check.errorf(jmp.Label, code, msg, name)
	}

	// spec: "It is illegal to define a label that is never used."
	for name, obj := range all.elems {
		obj = resolve(name, obj)
		if lbl := obj.(*Label); !lbl.used {
			check.softErrorf(lbl, UnusedLabel, "label %s declared and not used", quote(lbl.name))
		}
	}
}

// A block tracks label declarations in a block and its enclosing blocks.
type block struct {
	parent *block                      // enclosing block
	lstmt  *ast.LabeledStmt            // labeled statement to which this block belongs, or nil
	labels map[string]*ast.LabeledStmt // allocated lazily
}

// insert records a new label declaration for the current block.
// The label must not have been declared before in any block.
func (b *block) insert(s *ast.LabeledStmt) {
	name := s.Label.Name
	if debug {
		assert(b.gotoTarget(name) == nil)
	}
	labels := b.labels
	if labels == nil {
		labels = make(map[string]*ast.LabeledStmt)
		b.labels = labels
	}
	labels[name] = s
}

// gotoTarget returns the labeled statement in the current
// or an enclosing block with the given label name, or nil.
func (b *block) gotoTarget(name string) *ast.LabeledStmt {
	for s := b; s != nil; s = s.parent {
		if t := s.labels[name]; t != nil {
			return t
		}
	}
	return nil
}

// enclosingTarget returns the innermost enclosing labeled
// statement with the given label name, or nil.
func (b *block) enclosingTarget(name string) *ast.LabeledStmt {
	for s := b; s != nil; s = s.parent {
		if t := s.lstmt; t != nil && t.Label.Name == name {
			return t
		}
	}
	return nil
}

// blockBranches processes a block's statement list and returns the set of outgoing forward jumps.
// all is the scope of all declared labels, parent the set of labels declared in the immediately
// enclosing block, and lstmt is the labeled statement this block is associated with (or nil).
func (check *Checker) blockBranches(all *Scope, parent *block, lstmt *ast.LabeledStmt, list []ast.Stmt) []*ast.BranchStmt {
	b := &block{parent: parent, lstmt: lstmt}

	var (
		varDeclPos         token.Pos
		fwdJumps, badJumps []*ast.BranchStmt
	)

	// All forward jumps jumping over a variable declaration are possibly
	// invalid (they may still jump out of the block and be ok).
	// recordVarDecl records them for the given position.
	recordVarDecl := func(pos token.Pos) {
		varDeclPos = pos
		badJumps = append(badJumps[:0], fwdJumps...) // copy fwdJumps to badJumps
	}

	jumpsOverVarDecl := func(jmp *ast.BranchStmt) bool {
		if varDeclPos.IsValid() {
			for _, bad := range badJumps {
				if jmp == bad {
					return true
				}
			}
		}
		return false
	}

	blockBranches := func(lstmt *ast.LabeledStmt, list []ast.Stmt) {
		// Unresolved forward jumps inside the nested block
		// become forward jumps in the current block.
		fwdJumps = append(fwdJumps, check.blockBranches(all, b, lstmt, list)...)
	}

	var stmtBranches func(ast.Stmt)
	stmtBranches = func(s ast.Stmt) {
		switch s := s.(type) {
		case *ast.DeclStmt:
			if d, _ := s.Decl.(*ast.GenDecl); d != nil && d.Tok == token.VAR {
				recordVarDecl(d.Pos())
			}

		case *ast.LabeledStmt:
			// declare non-blank label
			if name := s.Label.Name; name != "_" {
				lbl := NewLabel(s.Label.Pos(), check.pkg, name)
				if alt := all.Insert(lbl); alt != nil {
					err := check.newError(DuplicateLabel)
					err.soft = true
					err.addf(lbl, "label %s already declared", quote(name))
					err.addAltDecl(alt)
					err.report()
					// ok to continue
				} else {
					b.insert(s)
					check.recordDef(s.Label, lbl)
				}
				// resolve matching forward jumps and remove them from fwdJumps
				i := 0
				for _, jmp := range fwdJumps {
					if jmp.Label.Name == name {
						// match
						lbl.used = true
						check.recordUse(jmp.Label, lbl)
						if jumpsOverVarDecl(jmp) {
							check.softErrorf(
								jmp.Label,
								JumpOverDecl,
								"goto %s jumps over variable declaration at line %d",
								name,
								check.fset.Position(varDeclPos).Line,
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

		case *ast.BranchStmt:
			if s.Label == nil {
				return // checked in 1st pass (check.stmt)
			}

			// determine and validate target
			name := s.Label.Name
			switch s.Tok {
			case token.BREAK:
				// spec: "If there is a label, it must be that of an enclosing
				// "for", "switch", or "select" statement, and that is the one
				// whose execution terminates."
				valid := false
				if t := b.enclosingTarget(name); t != nil {
					switch t.Stmt.(type) {
					case *ast.SwitchStmt, *ast.TypeSwitchStmt, *ast.SelectStmt, *ast.ForStmt, *ast.RangeStmt:
						valid = true
					}
				}
				if !valid {
					check.errorf(s.Label, MisplacedLabel, "invalid break label %s", quote(name))
					return
				}

			case token.CONTINUE:
				// spec: "If there is a label, it must be that of an enclosing
				// "for" statement, and that is the one whose execution advances."
				valid := false
				if t := b.enclosingTarget(name); t != nil {
					switch t.Stmt.(type) {
					case *ast.ForStmt, *ast.RangeStmt:
						valid = true
					}
				}
				if !valid {
					check.errorf(s.Label, MisplacedLabel, "invalid continue label %s", quote(name))
					return
				}

			case token.GOTO:
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

		case *ast.AssignStmt:
			if s.Tok == token.DEFINE {
				recordVarDecl(s.Pos())
			}

		case *ast.BlockStmt:
			blockBranches(lstmt, s.List)

		case *ast.IfStmt:
			stmtBranches(s.Body)
			if s.Else != nil {
				stmtBranches(s.Else)
			}

		case *ast.CaseClause:
			blockBranches(nil, s.Body)

		case *ast.SwitchStmt:
			stmtBranches(s.Body)

		case *ast.TypeSwitchStmt:
			stmtBranches(s.Body)

		case *ast.CommClause:
			blockBranches(nil, s.Body)

		case *ast.SelectStmt:
			stmtBranches(s.Body)

		case *ast.ForStmt:
			stmtBranches(s.Body)

		case *ast.RangeStmt:
			stmtBranches(s.Body)
		}
	}

	for _, s := range list {
		stmtBranches(s)
	}

	return fwdJumps
}
