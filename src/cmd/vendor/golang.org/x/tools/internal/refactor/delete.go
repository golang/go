// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package refactor

// This file defines operations for computing deletion edits.

import (
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"slices"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/ast/edge"
	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/internal/astutil"
	"golang.org/x/tools/internal/typesinternal"
	"golang.org/x/tools/internal/typesinternal/typeindex"
)

// DeleteVar returns edits to delete the declaration of a variable or
// constant whose defining identifier is curId.
//
// It handles variants including:
// - GenDecl > ValueSpec versus AssignStmt;
// - RHS expression has effects, or not;
// - entire statement/declaration may be eliminated;
// and removes associated comments.
//
// If it cannot make the necessary edits, such as for a function
// parameter or result, it returns nil.
func DeleteVar(tokFile *token.File, info *types.Info, curId inspector.Cursor) []analysis.TextEdit {
	switch ek, _ := curId.ParentEdge(); ek {
	case edge.ValueSpec_Names:
		return deleteVarFromValueSpec(tokFile, info, curId)

	case edge.AssignStmt_Lhs:
		return deleteVarFromAssignStmt(tokFile, info, curId)
	}

	// e.g. function receiver, parameter, or result,
	// or "switch v := expr.(T) {}" (which has no object).
	return nil
}

// deleteVarFromValueSpec returns edits to delete the declaration of a
// variable or constant within a ValueSpec.
//
// Precondition: curId is Ident beneath ValueSpec.Names beneath GenDecl.
//
// See also [deleteVarFromAssignStmt], which has parallel structure.
func deleteVarFromValueSpec(tokFile *token.File, info *types.Info, curIdent inspector.Cursor) []analysis.TextEdit {
	var (
		id      = curIdent.Node().(*ast.Ident)
		curSpec = curIdent.Parent()
		spec    = curSpec.Node().(*ast.ValueSpec)
	)

	declaresOtherNames := slices.ContainsFunc(spec.Names, func(name *ast.Ident) bool {
		return name != id && name.Name != "_"
	})
	noRHSEffects := !slices.ContainsFunc(spec.Values, func(rhs ast.Expr) bool {
		return !typesinternal.NoEffects(info, rhs)
	})
	if !declaresOtherNames && noRHSEffects {
		// The spec is no longer needed, either to declare
		// other variables, or for its side effects.
		return DeleteSpec(tokFile, curSpec)
	}

	// The spec is still needed, either for
	// at least one LHS, or for effects on RHS.
	// Blank out or delete just one LHS.

	_, index := curIdent.ParentEdge() // index of LHS within ValueSpec.Names

	// If there is no RHS, we can delete the LHS.
	if len(spec.Values) == 0 {
		var pos, end token.Pos
		if index == len(spec.Names)-1 {
			// Delete final name.
			//
			// var _, lhs1 T
			//      ------
			pos = spec.Names[index-1].End()
			end = spec.Names[index].End()
		} else {
			// Delete non-final name.
			//
			// var lhs0, _ T
			//     ------
			pos = spec.Names[index].Pos()
			end = spec.Names[index+1].Pos()
		}
		return []analysis.TextEdit{{
			Pos: pos,
			End: end,
		}}
	}

	// If the assignment is n:n and the RHS has no effects,
	// we can delete the LHS and its corresponding RHS.
	if len(spec.Names) == len(spec.Values) &&
		typesinternal.NoEffects(info, spec.Values[index]) {

		if index == len(spec.Names)-1 {
			// Delete final items.
			//
			// var _, lhs1 = rhs0, rhs1
			//      ------       ------
			return []analysis.TextEdit{
				{
					Pos: spec.Names[index-1].End(),
					End: spec.Names[index].End(),
				},
				{
					Pos: spec.Values[index-1].End(),
					End: spec.Values[index].End(),
				},
			}
		} else {
			// Delete non-final items.
			//
			// var lhs0, _ = rhs0, rhs1
			//     ------    ------
			return []analysis.TextEdit{
				{
					Pos: spec.Names[index].Pos(),
					End: spec.Names[index+1].Pos(),
				},
				{
					Pos: spec.Values[index].Pos(),
					End: spec.Values[index+1].Pos(),
				},
			}
		}
	}

	// We cannot delete the RHS.
	// Blank out the LHS.
	return []analysis.TextEdit{{
		Pos:     id.Pos(),
		End:     id.End(),
		NewText: []byte("_"),
	}}
}

// Precondition: curId is Ident beneath AssignStmt.Lhs.
//
// See also [deleteVarFromValueSpec], which has parallel structure.
func deleteVarFromAssignStmt(tokFile *token.File, info *types.Info, curIdent inspector.Cursor) []analysis.TextEdit {
	var (
		id      = curIdent.Node().(*ast.Ident)
		curStmt = curIdent.Parent()
		assign  = curStmt.Node().(*ast.AssignStmt)
	)

	declaresOtherNames := slices.ContainsFunc(assign.Lhs, func(lhs ast.Expr) bool {
		lhsId, ok := lhs.(*ast.Ident)
		return ok && lhsId != id && lhsId.Name != "_"
	})
	noRHSEffects := !slices.ContainsFunc(assign.Rhs, func(rhs ast.Expr) bool {
		return !typesinternal.NoEffects(info, rhs)
	})
	if !declaresOtherNames && noRHSEffects {
		// The assignment is no longer needed, either to
		// declare other variables, or for its side effects.
		if edits := DeleteStmt(tokFile, curStmt); edits != nil {
			return edits
		}
		// Statement could not not be deleted in this context.
		// Fall back to conservative deletion.
	}

	// The assign is still needed, either for
	// at least one LHS, or for effects on RHS,
	// or because it cannot deleted because of its context.
	// Blank out or delete just one LHS.

	// If the assignment is 1:1 and the RHS has no effects,
	// we can delete the LHS and its corresponding RHS.
	_, index := curIdent.ParentEdge()
	if len(assign.Lhs) > 1 &&
		len(assign.Lhs) == len(assign.Rhs) &&
		typesinternal.NoEffects(info, assign.Rhs[index]) {

		if index == len(assign.Lhs)-1 {
			// Delete final items.
			//
			// _, lhs1 := rhs0, rhs1
			//  ------        ------
			return []analysis.TextEdit{
				{
					Pos: assign.Lhs[index-1].End(),
					End: assign.Lhs[index].End(),
				},
				{
					Pos: assign.Rhs[index-1].End(),
					End: assign.Rhs[index].End(),
				},
			}
		} else {
			// Delete non-final items.
			//
			// lhs0, _ := rhs0, rhs1
			// ------     ------
			return []analysis.TextEdit{
				{
					Pos: assign.Lhs[index].Pos(),
					End: assign.Lhs[index+1].Pos(),
				},
				{
					Pos: assign.Rhs[index].Pos(),
					End: assign.Rhs[index+1].Pos(),
				},
			}
		}
	}

	// We cannot delete the RHS.
	// Blank out the LHS.
	edits := []analysis.TextEdit{{
		Pos:     id.Pos(),
		End:     id.End(),
		NewText: []byte("_"),
	}}

	// If this eliminates the final variable declared by
	// an := statement, we need to turn it into an =
	// assignment to avoid a "no new variables on left
	// side of :=" error.
	if !declaresOtherNames {
		edits = append(edits, analysis.TextEdit{
			Pos:     assign.TokPos,
			End:     assign.TokPos + token.Pos(len(":=")),
			NewText: []byte("="),
		})
	}

	return edits
}

// DeleteSpec returns edits to delete the {Type,Value}Spec identified by curSpec.
//
// TODO(adonovan): add test suite. Test for consts as well.
func DeleteSpec(tokFile *token.File, curSpec inspector.Cursor) []analysis.TextEdit {
	var (
		spec    = curSpec.Node().(ast.Spec)
		curDecl = curSpec.Parent()
		decl    = curDecl.Node().(*ast.GenDecl)
	)

	// If it is the sole spec in the decl,
	// delete the entire decl.
	if len(decl.Specs) == 1 {
		return DeleteDecl(tokFile, curDecl)
	}

	// Delete the spec and its comments.
	_, index := curSpec.ParentEdge() // index of ValueSpec within GenDecl.Specs
	pos, end := spec.Pos(), spec.End()
	if doc := astutil.DocComment(spec); doc != nil {
		pos = doc.Pos() // leading comment
	}
	if index == len(decl.Specs)-1 {
		// Delete final spec.
		if c := eolComment(spec); c != nil {
			//  var (v int // comment \n)
			end = c.End()
		}
	} else {
		// Delete non-final spec.
		//   var ( a T; b T )
		//         -----
		end = decl.Specs[index+1].Pos()
	}
	return []analysis.TextEdit{{
		Pos: pos,
		End: end,
	}}
}

// DeleteDecl returns edits to delete the ast.Decl identified by curDecl.
//
// TODO(adonovan): add test suite.
func DeleteDecl(tokFile *token.File, curDecl inspector.Cursor) []analysis.TextEdit {
	decl := curDecl.Node().(ast.Decl)

	ek, _ := curDecl.ParentEdge()
	switch ek {
	case edge.DeclStmt_Decl:
		return DeleteStmt(tokFile, curDecl.Parent())

	case edge.File_Decls:
		pos, end := decl.Pos(), decl.End()
		if doc := astutil.DocComment(decl); doc != nil {
			pos = doc.Pos()
		}

		// Delete free-floating comments on same line as rparen.
		//    var (...) // comment
		var (
			file        = curDecl.Parent().Node().(*ast.File)
			lineOf      = tokFile.Line
			declEndLine = lineOf(decl.End())
		)
		for _, cg := range file.Comments {
			for _, c := range cg.List {
				if c.Pos() < end {
					continue // too early
				}
				commentEndLine := lineOf(c.End())
				if commentEndLine > declEndLine {
					break // too late
				} else if lineOf(c.Pos()) == declEndLine && commentEndLine == declEndLine {
					end = c.End()
				}
			}
		}

		return []analysis.TextEdit{{
			Pos: pos,
			End: end,
		}}

	default:
		panic(fmt.Sprintf("Decl parent is %v, want DeclStmt or File", ek))
	}
}

// DeleteStmt returns the edits to remove the [ast.Stmt] identified by
// curStmt, if it is contained within a BlockStmt, CaseClause,
// CommClause, or is the STMT in switch STMT; ... {...}. It returns nil otherwise.
func DeleteStmt(tokFile *token.File, curStmt inspector.Cursor) []analysis.TextEdit {
	stmt := curStmt.Node().(ast.Stmt)
	// if the stmt is on a line by itself delete the whole line
	// otherwise just delete the statement.

	// this logic would be a lot simpler with the file contents, and somewhat simpler
	// if the cursors included the comments.

	lineOf := tokFile.Line
	stmtStartLine, stmtEndLine := lineOf(stmt.Pos()), lineOf(stmt.End())

	var from, to token.Pos
	// bounds of adjacent syntax/comments on same line, if any
	limits := func(left, right token.Pos) {
		if lineOf(left) == stmtStartLine {
			from = left
		}
		if lineOf(right) == stmtEndLine {
			to = right
		}
	}
	// TODO(pjw): there are other places a statement might be removed:
	// IfStmt = "if" [ SimpleStmt ";" ] Expression Block [ "else" ( IfStmt | Block ) ] .
	// (removing the blocks requires more rewriting than this routine would do)
	// CommCase   = "case" ( SendStmt | RecvStmt ) | "default" .
	// (removing the stmt requires more rewriting, and it's unclear what the user means)
	switch parent := curStmt.Parent().Node().(type) {
	case *ast.SwitchStmt:
		limits(parent.Switch, parent.Body.Lbrace)
	case *ast.TypeSwitchStmt:
		limits(parent.Switch, parent.Body.Lbrace)
		if parent.Assign == stmt {
			return nil // don't let the user break the type switch
		}
	case *ast.BlockStmt:
		limits(parent.Lbrace, parent.Rbrace)
	case *ast.CommClause:
		limits(parent.Colon, curStmt.Parent().Parent().Node().(*ast.BlockStmt).Rbrace)
		if parent.Comm == stmt {
			return nil // maybe the user meant to remove the entire CommClause?
		}
	case *ast.CaseClause:
		limits(parent.Colon, curStmt.Parent().Parent().Node().(*ast.BlockStmt).Rbrace)
	case *ast.ForStmt:
		limits(parent.For, parent.Body.Lbrace)

	default:
		return nil // not one of ours
	}

	if prev, found := curStmt.PrevSibling(); found && lineOf(prev.Node().End()) == stmtStartLine {
		from = prev.Node().End() // preceding statement ends on same line
	}
	if next, found := curStmt.NextSibling(); found && lineOf(next.Node().Pos()) == stmtEndLine {
		to = next.Node().Pos() // following statement begins on same line
	}
	// and now for the comments
Outer:
	for _, cg := range astutil.EnclosingFile(curStmt).Comments {
		for _, co := range cg.List {
			if lineOf(co.End()) < stmtStartLine {
				continue
			} else if lineOf(co.Pos()) > stmtEndLine {
				break Outer // no more are possible
			}
			if lineOf(co.End()) == stmtStartLine && co.End() < stmt.Pos() {
				if !from.IsValid() || co.End() > from {
					from = co.End()
					continue // maybe there are more
				}
			}
			if lineOf(co.Pos()) == stmtEndLine && co.Pos() > stmt.End() {
				if !to.IsValid() || co.Pos() < to {
					to = co.Pos()
					continue // maybe there are more
				}
			}
		}
	}
	// if either from or to is valid, just remove the statement
	// otherwise remove the line
	edit := analysis.TextEdit{Pos: stmt.Pos(), End: stmt.End()}
	if from.IsValid() || to.IsValid() {
		// remove just the statement.
		// we can't tell if there is a ; or whitespace right after the statement
		// ideally we'd like to remove the former and leave the latter
		// (if gofmt has run, there likely won't be a ;)
		// In type switches we know there's a semicolon somewhere after the statement,
		// but the extra work for this special case is not worth it, as gofmt will fix it.
		return []analysis.TextEdit{edit}
	}
	// remove the whole line
	for lineOf(edit.Pos) == stmtStartLine {
		edit.Pos--
	}
	edit.Pos++ // get back tostmtStartLine
	for lineOf(edit.End) == stmtEndLine {
		edit.End++
	}
	return []analysis.TextEdit{edit}
}

// DeleteUnusedVars computes the edits required to delete the
// declarations of any local variables whose last uses are in the
// curDelend subtree, which is about to be deleted.
func DeleteUnusedVars(index *typeindex.Index, info *types.Info, tokFile *token.File, curDelend inspector.Cursor) []analysis.TextEdit {
	// TODO(adonovan): we might want to generalize this by
	// splitting the two phases below, so that we can gather
	// across a whole sequence of deletions then finally compute the
	// set of variables that are no longer wanted.

	// Count number of deletions of each var.
	delcount := make(map[*types.Var]int)
	for curId := range curDelend.Preorder((*ast.Ident)(nil)) {
		id := curId.Node().(*ast.Ident)
		if v, ok := info.Uses[id].(*types.Var); ok &&
			typesinternal.GetVarKind(v) == typesinternal.LocalVar { // always false before go1.25
			delcount[v]++
		}
	}

	// Delete declaration of each var that became unused.
	var edits []analysis.TextEdit
	for v, count := range delcount {
		if len(slices.Collect(index.Uses(v))) == count {
			if curDefId, ok := index.Def(v); ok {
				edits = append(edits, DeleteVar(tokFile, info, curDefId)...)
			}
		}
	}
	return edits
}

func eolComment(n ast.Node) *ast.CommentGroup {
	// TODO(adonovan): support:
	//    func f() {...} // comment
	switch n := n.(type) {
	case *ast.GenDecl:
		if !n.TokPos.IsValid() && len(n.Specs) == 1 {
			return eolComment(n.Specs[0])
		}
	case *ast.ValueSpec:
		return n.Comment
	case *ast.TypeSpec:
		return n.Comment
	}
	return nil
}
