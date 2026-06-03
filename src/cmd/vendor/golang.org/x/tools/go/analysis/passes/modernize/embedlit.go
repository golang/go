// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modernize

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"slices"
	"strings"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/edge"
	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/internal/analysis/analyzerutil"
	typeindexanalyzer "golang.org/x/tools/internal/analysis/typeindex"
	"golang.org/x/tools/internal/astutil"
	"golang.org/x/tools/internal/moreiters"
	"golang.org/x/tools/internal/typesinternal/typeindex"
	"golang.org/x/tools/internal/versions"
)

var EmbedLitAnalyzer = &analysis.Analyzer{
	Name: "embedlit",
	Doc:  analyzerutil.MustExtractDoc(doc, "embedlit"),
	Requires: []*analysis.Analyzer{
		inspect.Analyzer,
		typeindexanalyzer.Analyzer,
	},
	Run: runEmbedLit,
	URL: "https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/modernize#embedlit",
}

// Go1.27 introduced the ability to directly access embedded struct fields.
// The embedlit modernizer suggests two types of fixes that use this feature:
// 1. Removing redundant field type specifiers in embedded struct fields.
// 2. Moving embedded struct field assignments inside of the struct literal
// initialization.
func runEmbedLit(pass *analysis.Pass) (any, error) {
	var (
		inspect = pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)
		index   = pass.ResultOf[typeindexanalyzer.Analyzer].(*typeindex.Index)
		info    = pass.TypesInfo
	)
	for curLit := range inspect.Root().Preorder((*ast.CompositeLit)(nil)) {
		if curLit.ParentEdgeKind() != edge.KeyValueExpr_Value { // non-nested comp lit
			// TODO(mkalil): Figure out how to handle addition/removal of commas in
			// the comp lit when we observe code where both patterns apply. (This will
			// likely require a significant amount of work). For now, only apply edits
			// from one pattern at a time.
			if !embedlitUnnest(pass, info, curLit) {
				err := embedlitCombine(pass, index, info, curLit) // calls pass.ReadFile
				if err != nil {
					return nil, err
				}
			}
		}
	}
	return nil, nil
}

// Pattern A: removing unneeded embedded field type specifier from the struct
// literal.
// T{U: U{f: v, ...}} => T{f: v, ...}
// It returns true if it reported a diagnostic with edits.
func embedlitUnnest(pass *analysis.Pass, info *types.Info, curLit inspector.Cursor) bool {
	var (
		edits       []analysis.TextEdit
		names       []string // names of the embedded field types that can be removed
		lit         = curLit.Node().(*ast.CompositeLit)
		compLitType = info.TypeOf(lit)
	)

	// checkLit determines whether any of the fields in the given struct literal can
	// be promoted, and calculates the corresponding edits.
	var checkLit func(lit *ast.CompositeLit)
	checkLit = func(lit *ast.CompositeLit) {
		for _, elt := range lit.Elts {
			// Can't promote an unkeyed field; would result in a syntax error.
			if kv, ok := elt.(*ast.KeyValueExpr); ok {
				if innerLit := isEmbeddedFieldLit(info, compLitType, kv); innerLit != nil {
					// Emit edits to delete the unnecessary embedded field type specifier
					// and its closing brace.
					closingPos := innerLit.Rbrace
					if len(innerLit.Elts) > 0 {
						// Delete any inner trailing commas or white space. Extra trailing commas
						// would result in invalid code.
						closingPos = innerLit.Elts[len(innerLit.Elts)-1].End()
					}
					file := astutil.EnclosingFile(curLit)
					// Enable modernizer only for Go1.27.
					if !analyzerutil.FileUsesGoVersion(pass, file, versions.Go1_27) {
						return
					}
					// If any comments overlap with the range to delete, don't suggest a fix.
					if !moreiters.Empty(astutil.Comments(file, kv.Pos(), innerLit.Lbrace+1)) ||
						!moreiters.Empty(astutil.Comments(file, closingPos, innerLit.Rbrace+1)) {
						continue
					}
					edits = append(edits, []analysis.TextEdit{
						// T{U: U{f: v, ...}}
						//   -----         -
						{
							// Delete the key and the opening brace of the inner struct literal.
							Pos: kv.Pos(),
							End: innerLit.Lbrace + 1,
						},
						{
							// Delete the corresponding closing brace, including preceding
							// white space or commas. Failing to delete trailing commas may
							// result in invalid code.
							Pos: closingPos,
							End: innerLit.Rbrace + 1,
						},
					}...)
					names = append(names, kv.Key.(*ast.Ident).Name)
					checkLit(innerLit)
				}
			}
		}
	}
	checkLit(lit)
	if len(edits) > 0 {
		pass.Report(analysis.Diagnostic{
			Pos:     curLit.Node().Pos(),
			End:     curLit.Node().End(),
			Message: "embedded field type can be removed from struct literal",
			SuggestedFixes: []analysis.SuggestedFix{
				{
					Message:   fmt.Sprintf("Remove embedded field type%s %s", cond(len(names) == 1, "", "s"), strings.Join(names, ", ")),
					TextEdits: edits,
				},
			},
		})
		return true
	}
	return false
}

// Pattern B: moving embedded field assignments inside the struct literal
// initialization.
// t := T{...}; t.x = x => t := T{..., x: x}
// (or var t = ...)
func embedlitCombine(pass *analysis.Pass, index *typeindex.Index, info *types.Info, curLit inspector.Cursor) error {
	compLit := curLit.Node().(*ast.CompositeLit)
	if !moreiters.Every(slices.Values(compLit.Elts), func(e ast.Expr) bool {
		return is[*ast.KeyValueExpr](e)
	}) {
		// Promoting additional embedded fields would result in mixing keyed and
		// unkeyed fields, which isn't allowed.
		return nil
	}
	var (
		// Ident for "t" in the assignment.
		lhs *ast.Ident
		// The cursor representing the statement that initializes the comp lit "t".
		// We use its siblings to search for field assignments and verify that there
		// are no intervening statements, in case those statements observe "t".
		curStmt inspector.Cursor
	)
	switch curLit.ParentEdgeKind() {
	case edge.AssignStmt_Rhs:
		assign := curLit.Parent().Node().(*ast.AssignStmt)
		// TODO(mkalil): Handle lhs forms that aren't idents, i.e. x.y[i] = T{...}.
		if id, ok := assign.Lhs[curLit.ParentEdgeIndex()].(*ast.Ident); ok {
			lhs = id
			curStmt = curLit.Parent()
		}
	case edge.ValueSpec_Values:
		spec := curLit.Parent().Node().(*ast.ValueSpec)
		lhs = spec.Names[curLit.ParentEdgeIndex()]
		if decl, ok := moreiters.First(curLit.Enclosing((*ast.DeclStmt)(nil))); ok {
			curStmt = decl
		}
	default:
		return nil
	}

	if lhs == nil || !curStmt.Valid() {
		return nil
	}

	var (
		tObj = info.ObjectOf(lhs)
		// Marks the contiguous block of embedded field assign statements that will
		// be moved into the struct initialization.
		firstStmt, lastStmt inspector.Cursor
	)
stmtloop:
	for {
		var ok bool
		curStmt, ok = curStmt.NextSibling()
		if !ok {
			break // end of (e.g.) block
		}
		// All embedded field value assignments must immediately follow the struct
		// initialization.
		assign, ok := curStmt.Node().(*ast.AssignStmt)
		if !ok || len(assign.Lhs) != 1 || !(assign.Tok == token.ASSIGN || assign.Tok == token.DEFINE) {
			// TODO(mkalil): handle multi-assignments like t.x, t.y = 1, 2
			break
		}
		expr := assign.Lhs[0]
		sel, ok := expr.(*ast.SelectorExpr)
		if !ok {
			break
		}
		// Verify that sel.X refers to the same object as "t"
		selXId, ok := sel.X.(*ast.Ident)
		if !ok {
			// TODO(mkalil): handle deeply nested expressions like t.B.x
			break
		}
		obj := info.ObjectOf(selXId)
		if obj != tObj {
			break
		}
		rhsCur := curStmt.ChildAt(edge.AssignStmt_Rhs, 0)
		if uses(index, rhsCur, tObj) {
			break
		}
		for c := range rhsCur.Preorder((*ast.Ident)(nil)) {
			id := c.Node().(*ast.Ident)
			// If the rhs uses a value of t (e.g. t.x = t.y), don't suggest a fix because
			// we can't evaluate t.y when constructing the new literal.
			if info.ObjectOf(id) == tObj {
				break stmtloop
			}
			// Note: we don't need to worry about expressions with side effects
			// changing the behavior when moved inside the comp lit. The order of
			// effects will be preserved because we preserve the order of the key
			// value pairs inside the comp lit.
		}
		if !firstStmt.Valid() {
			firstStmt = curStmt
		}
		lastStmt = curStmt
	}

	if !firstStmt.Valid() {
		return nil
	}

	file := astutil.EnclosingFile(curLit)
	// Enable modernizer only for Go1.27.
	if !analyzerutil.FileUsesGoVersion(pass, file, versions.Go1_27) {
		return nil
	}

	// Read file content to determine if the struct lit has a trailing comma
	// after its last element.
	tokFile := pass.Fset.File(compLit.Rbrace)
	filename := tokFile.Name()
	src, err := pass.ReadFile(filename)
	if err != nil {
		return err
	}

	hasTrailingComma := false
	if len(compLit.Elts) > 0 {
		lastElt := compLit.Elts[len(compLit.Elts)-1]
		lastEltOffset := tokFile.Offset(lastElt.End())
		rbraceOffset := tokFile.Offset(compLit.Rbrace)
		hasTrailingComma = bytes.Contains(src[lastEltOffset:rbraceOffset], []byte(","))
	}
	var edits []analysis.TextEdit
	// Emit edits to move the field assignment into the struct lit while
	// removing it from its current place.
	// t := T{...}; t.x = v
	//           ----- --- -
	// t := T{...,    x:  v}

	// Add a trailing comma before the closing brace of compLit if one doesn't
	// exist, and delete the closing brace itself.
	// t := T{...}; t.x = v
	//           -
	// t := T{..., t.x = v
	if len(compLit.Elts) > 0 && !hasTrailingComma {
		edits = append(edits, analysis.TextEdit{
			Pos:     compLit.Rbrace,
			End:     compLit.Rbrace + 1,
			NewText: []byte(","),
		})
	} else {
		edits = append(edits, analysis.TextEdit{
			Pos: compLit.Rbrace,
			End: compLit.Rbrace + 1,
		})
	}

	// For each assignment:
	// t.x = v
	// -- ---
	//   x : v
	curStmt = firstStmt
	var prevStmt inspector.Cursor
	for {
		assign := curStmt.Node().(*ast.AssignStmt)
		expr := assign.Lhs[0]
		sel := expr.(*ast.SelectorExpr)
		// Delete "t."
		edits = append(edits, analysis.TextEdit{
			Pos: assign.Pos(),
			End: sel.Sel.Pos(),
		})
		// Replace "=" with ":"
		edits = append(edits, analysis.TextEdit{
			Pos:     expr.End(),
			End:     assign.TokPos + 1,
			NewText: []byte(":"),
		})

		// Add a comma after the previous assignment if this is not the first one.
		if prevStmt.Valid() {
			edits = append(edits, analysis.TextEdit{
				Pos:     prevStmt.Node().End(),
				NewText: []byte(","),
			})
		}

		// For the last assignment, add the closing brace of the struct lit.
		if curStmt == lastStmt {
			edits = append(edits, analysis.TextEdit{
				Pos:     assign.End(),
				NewText: []byte("}"),
			})
			break
		}
		prevStmt = curStmt
		curStmt, _ = curStmt.NextSibling() // can't fail because we break out of the loop when we hit lastStmt
	}

	pass.Report(analysis.Diagnostic{
		Pos:     curLit.Node().Pos(),
		End:     curLit.Node().End(),
		Message: "embedded field assignment can be moved to struct literal",
		SuggestedFixes: []analysis.SuggestedFix{
			{
				Message:   "Move embedded field assignment to struct literal",
				TextEdits: edits,
			},
		},
	})
	return nil
}

// isEmbeddedFieldLit determines whether elt is a KeyValueExpr "T: T{...}" for
// an embedded field for which we can safely remove its type.
// If so, it returns the corresponding CompositeLit.
// If elt contains an unkeyed field or ambiguous type, it returns nil.
func isEmbeddedFieldLit(info *types.Info, topLevelType types.Type, kv *ast.KeyValueExpr) *ast.CompositeLit {
	obj := keyedField(info, kv)
	if obj == nil || !obj.Embedded() {
		return nil
	}
	lit, ok := kv.Value.(*ast.CompositeLit)
	if !ok || len(lit.Elts) == 0 {
		// Skip if the struct literal is empty.
		return nil
	}
	// We cannot remove this type if any of its nested composite elements have
	// unkeyed fields or are ambiguous, so we check for those conditions before
	// returning.
	for _, elt := range lit.Elts {
		kv, ok := elt.(*ast.KeyValueExpr)
		if !ok {
			return nil
		}
		obj := keyedField(info, kv)
		if obj == nil {
			return nil
		}
		k := kv.Key.(*ast.Ident) // can't fail
		// Cannot promote an ambiguous type, for example:
		// type T struct { A; B }
		// type A struct { x int }
		// type B struct { x int }
		// _ = T{A: A{x: 1}}
		// cannot be simplified to T{x: 1} because T has two different embedded fields called "x".
		// We also reject composite literals with slice elements, as parentObj will be nil.
		parentObj, _, _ := types.LookupFieldOrMethod(topLevelType, true, obj.Pkg(), k.Name)
		if parentObj != obj {
			return nil
		}
	}
	return lit
}

// keyedField reports whether the key of kv is an embedded field type. If so, it
// returns the type of the embedded field, otherwise it returns nil.
func keyedField(info *types.Info, kv *ast.KeyValueExpr) *types.Var {
	k, ok := kv.Key.(*ast.Ident)
	if !ok {
		return nil
	}
	obj, ok := info.ObjectOf(k).(*types.Var)
	if !ok || !obj.IsField() {
		return nil
	}
	return obj
}
