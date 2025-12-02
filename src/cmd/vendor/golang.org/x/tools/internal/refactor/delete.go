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
func DeleteVar(tokFile *token.File, info *types.Info, curId inspector.Cursor) []Edit {
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
func deleteVarFromValueSpec(tokFile *token.File, info *types.Info, curIdent inspector.Cursor) []Edit {
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
		return []Edit{{
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
			return []Edit{
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
			return []Edit{
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
	return []Edit{{
		Pos:     id.Pos(),
		End:     id.End(),
		NewText: []byte("_"),
	}}
}

// Precondition: curId is Ident beneath AssignStmt.Lhs.
//
// See also [deleteVarFromValueSpec], which has parallel structure.
func deleteVarFromAssignStmt(tokFile *token.File, info *types.Info, curIdent inspector.Cursor) []Edit {
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
			return []Edit{
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
			return []Edit{
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
	edits := []Edit{{
		Pos:     id.Pos(),
		End:     id.End(),
		NewText: []byte("_"),
	}}

	// If this eliminates the final variable declared by
	// an := statement, we need to turn it into an =
	// assignment to avoid a "no new variables on left
	// side of :=" error.
	if !declaresOtherNames {
		edits = append(edits, Edit{
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
func DeleteSpec(tokFile *token.File, curSpec inspector.Cursor) []Edit {
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
	return []Edit{{
		Pos: pos,
		End: end,
	}}
}

// DeleteDecl returns edits to delete the ast.Decl identified by curDecl.
//
// TODO(adonovan): add test suite.
func DeleteDecl(tokFile *token.File, curDecl inspector.Cursor) []Edit {
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

		return []Edit{{
			Pos: pos,
			End: end,
		}}

	default:
		panic(fmt.Sprintf("Decl parent is %v, want DeclStmt or File", ek))
	}
}

// find leftmost Pos bigger than start and rightmost less than end
func filterPos(nds []*ast.Comment, start, end token.Pos) (token.Pos, token.Pos, bool) {
	l, r := end, token.NoPos
	ok := false
	for _, n := range nds {
		if n.Pos() > start && n.Pos() < l {
			l = n.Pos()
			ok = true
		}
		if n.End() <= end && n.End() > r {
			r = n.End()
			ok = true
		}
	}
	return l, r, ok
}

// DeleteStmt returns the edits to remove the [ast.Stmt] identified by
// curStmt if it recognizes the context. It returns nil otherwise.
// TODO(pjw, adonovan): it should not return nil, it should return an error
//
// DeleteStmt is called with just the AST so it has trouble deciding if
// a comment is associated with the statement to be deleted. For instance,
//
//	for /*A*/ init()/*B*/;/*C/cond()/*D/;/*E*/post() /*F*/ { /*G*/}
//
// comment B and C are indistinguishable, as are D and E. That is, as the
// AST does not say where the semicolons are, B and C could go either
// with the init() or the cond(), so cannot be removed safely. The same
// is true for D, E, and the post(). (And there are other similar cases.)
// But the other comments can be removed as they are unambiguously
// associated with the statement being deleted. In particular,
// it removes whole lines like
//
//	stmt // comment
func DeleteStmt(file *token.File, curStmt inspector.Cursor) []Edit {
	// if the stmt is on a line by itself, or a range of lines, delete the whole thing
	// including comments. Except for the heads of switches, type
	// switches, and for-statements that's the usual case. Complexity occurs where
	// there are multiple statements on the same line, and adjacent comments.

	// In that case we remove some adjacent comments:
	// In me()/*A*/;b(), comment A cannot be removed, because the ast
	// is indistinguishable from me();/*A*/b()
	// and the same for cases like switch me()/*A*/; x.(type) {

	// this would be more precise with the file contents, or if the ast
	// contained the location of semicolons
	var (
		stmt          = curStmt.Node().(ast.Stmt)
		tokFile       = file
		lineOf        = tokFile.Line
		stmtStartLine = lineOf(stmt.Pos())
		stmtEndLine   = lineOf(stmt.End())

		leftSyntax, rightSyntax     token.Pos      // pieces of parent node on stmt{Start,End}Line
		leftComments, rightComments []*ast.Comment // comments before/after stmt on the same line
	)

	// remember the Pos that are on the same line as stmt
	use := func(left, right token.Pos) {
		if lineOf(left) == stmtStartLine {
			leftSyntax = left
		}
		if lineOf(right) == stmtEndLine {
			rightSyntax = right
		}
	}

	// find the comments, if any, on the same line
Big:
	for _, cg := range astutil.EnclosingFile(curStmt).Comments {
		for _, co := range cg.List {
			if lineOf(co.End()) < stmtStartLine {
				continue
			} else if lineOf(co.Pos()) > stmtEndLine {
				break Big // no more are possible
			}
			if lineOf(co.End()) == stmtStartLine && co.End() <= stmt.Pos() {
				// comment is before the statement
				leftComments = append(leftComments, co)
			} else if lineOf(co.Pos()) == stmtEndLine && co.Pos() >= stmt.End() {
				// comment is after the statement
				rightComments = append(rightComments, co)
			}
		}
	}

	// find any other syntax on the same line
	var (
		leftStmt, rightStmt token.Pos // end/start positions of sibling statements in a []Stmt list
		inStmtList          = false
		curParent           = curStmt.Parent()
	)
	switch parent := curParent.Node().(type) {
	case *ast.BlockStmt:
		use(parent.Lbrace, parent.Rbrace)
		inStmtList = true
	case *ast.CaseClause:
		use(parent.Colon, curStmt.Parent().Parent().Node().(*ast.BlockStmt).Rbrace)
		inStmtList = true
	case *ast.CommClause:
		if parent.Comm == stmt {
			return nil // maybe the user meant to remove the entire CommClause?
		}
		use(parent.Colon, curStmt.Parent().Parent().Node().(*ast.BlockStmt).Rbrace)
		inStmtList = true
	case *ast.ForStmt:
		use(parent.For, parent.Body.Lbrace)
		// special handling, as init;cond;post BlockStmt is not a statment list
		if parent.Init != nil && parent.Cond != nil && stmt == parent.Init && lineOf(parent.Cond.Pos()) == lineOf(stmt.End()) {
			rightStmt = parent.Cond.Pos()
		} else if parent.Post != nil && parent.Cond != nil && stmt == parent.Post && lineOf(parent.Cond.End()) == lineOf(stmt.Pos()) {
			leftStmt = parent.Cond.End()
		}
	case *ast.IfStmt:
		switch stmt {
		case parent.Init:
			use(parent.If, parent.Body.Lbrace)
		case parent.Else:
			// stmt is the {...} in "if cond {} else {...}" and removing
			// it would require removing the 'else' keyword, but the ast
			// does not contain its position.
			return nil
		}
	case *ast.SwitchStmt:
		use(parent.Switch, parent.Body.Lbrace)
	case *ast.TypeSwitchStmt:
		if stmt == parent.Assign {
			return nil // don't remove .(type)
		}
		use(parent.Switch, parent.Body.Lbrace)
	default:
		return nil // not one of ours
	}

	if inStmtList {
		// find the siblings, if any, on the same line
		if prev, found := curStmt.PrevSibling(); found && lineOf(prev.Node().End()) == stmtStartLine {
			if _, ok := prev.Node().(ast.Stmt); ok {
				leftStmt = prev.Node().End() // preceding statement ends on same line
			}
		}
		if next, found := curStmt.NextSibling(); found && lineOf(next.Node().Pos()) == stmtEndLine {
			rightStmt = next.Node().Pos() // following statement begins on same line
		}
	}

	// compute the left and right limits of the edit
	var leftEdit, rightEdit token.Pos
	if leftStmt.IsValid() {
		leftEdit = stmt.Pos() // can't remove preceding comments: a()/*A*/; me()
	} else if leftSyntax.IsValid() {
		// remove intervening leftComments
		if a, _, ok := filterPos(leftComments, leftSyntax, stmt.Pos()); ok {
			leftEdit = a
		} else {
			leftEdit = stmt.Pos()
		}
	} else { // remove whole line
		for leftEdit = stmt.Pos(); lineOf(leftEdit) == stmtStartLine; leftEdit-- {
		}
		if leftEdit < stmt.Pos() {
			leftEdit++ // beginning of line
		}
	}
	if rightStmt.IsValid() {
		rightEdit = stmt.End() // can't remove following comments
	} else if rightSyntax.IsValid() {
		// remove intervening rightComments
		if _, b, ok := filterPos(rightComments, stmt.End(), rightSyntax); ok {
			rightEdit = b
		} else {
			rightEdit = stmt.End()
		}
	} else { // remove whole line
		fend := token.Pos(file.Base()) + token.Pos(file.Size())
		for rightEdit = stmt.End(); fend >= rightEdit && lineOf(rightEdit) == stmtEndLine; rightEdit++ {
		}
		// don't remove \n if there was other stuff earlier
		if leftSyntax.IsValid() || leftStmt.IsValid() {
			rightEdit--
		}
	}

	return []Edit{{Pos: leftEdit, End: rightEdit}}
}

// DeleteUnusedVars computes the edits required to delete the
// declarations of any local variables whose last uses are in the
// curDelend subtree, which is about to be deleted.
func DeleteUnusedVars(index *typeindex.Index, info *types.Info, tokFile *token.File, curDelend inspector.Cursor) []Edit {
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
	var edits []Edit
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
