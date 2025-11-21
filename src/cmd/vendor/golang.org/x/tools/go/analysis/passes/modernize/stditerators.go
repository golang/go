// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modernize

import (
	"fmt"
	"go/ast"
	"go/token"
	"go/types"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/ast/edge"
	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/go/types/typeutil"
	"golang.org/x/tools/internal/analysis/analyzerutil"
	typeindexanalyzer "golang.org/x/tools/internal/analysis/typeindex"
	"golang.org/x/tools/internal/astutil"
	"golang.org/x/tools/internal/goplsexport"
	"golang.org/x/tools/internal/refactor"
	"golang.org/x/tools/internal/stdlib"
	"golang.org/x/tools/internal/typesinternal/typeindex"
)

var stditeratorsAnalyzer = &analysis.Analyzer{
	Name: "stditerators",
	Doc:  analyzerutil.MustExtractDoc(doc, "stditerators"),
	Requires: []*analysis.Analyzer{
		typeindexanalyzer.Analyzer,
	},
	Run: stditerators,
	URL: "https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/modernize#stditerators",
}

func init() {
	// Export to gopls until this is a published modernizer.
	goplsexport.StdIteratorsModernizer = stditeratorsAnalyzer
}

// stditeratorsTable records std types that have legacy T.{Len,At}
// iteration methods as well as a newer T.All method that returns an
// iter.Seq.
var stditeratorsTable = [...]struct {
	pkgpath, typename, lenmethod, atmethod, itermethod, elemname string
}{
	// Example: in go/types, (*Tuple).Variables returns an
	// iterator that replaces a loop over (*Tuple).{Len,At}.
	// The loop variable is named "v".
	{"go/types", "Interface", "NumEmbeddeds", "EmbeddedType", "EmbeddedTypes", "etyp"},
	{"go/types", "Interface", "NumExplicitMethods", "ExplicitMethod", "ExplicitMethods", "method"},
	{"go/types", "Interface", "NumMethods", "Method", "Methods", "method"},
	{"go/types", "MethodSet", "Len", "At", "Methods", "method"},
	{"go/types", "Named", "NumMethods", "Method", "Methods", "method"},
	{"go/types", "Scope", "NumChildren", "Child", "Children", "child"},
	{"go/types", "Struct", "NumFields", "Field", "Fields", "field"},
	{"go/types", "Tuple", "Len", "At", "Variables", "v"},
	{"go/types", "TypeList", "Len", "At", "Types", "t"},
	{"go/types", "TypeParamList", "Len", "At", "TypeParams", "tparam"},
	{"go/types", "Union", "Len", "Term", "Terms", "term"},
	// TODO(adonovan): support Seq2. Bonus: transform uses of both key and value.
	// {"reflect", "Value", "NumFields", "Field", "Fields", "field"},
}

// stditerators suggests fixes to replace loops using Len/At-style
// iterator APIs by a range loop over an iterator. The set of
// participating types and methods is defined by [iteratorsTable].
//
// Pattern:
//
//	for i := 0; i < x.Len(); i++ {
//		use(x.At(i))
//	}
//
// =>
//
//	for elem := range x.All() {
//		use(elem)
//	}
//
// Variant:
//
//	for i := range x.Len() { ... }
//
// Note: Iterators have a dynamic cost. How do we know that
// the user hasn't intentionally chosen not to use an
// iterator for that reason? We don't want to go fix to
// undo optimizations. Do we need a suppression mechanism?
func stditerators(pass *analysis.Pass) (any, error) {
	var (
		index = pass.ResultOf[typeindexanalyzer.Analyzer].(*typeindex.Index)
		info  = pass.TypesInfo
	)

	for _, row := range stditeratorsTable {
		// Don't offer fixes within the package
		// that defines the iterator in question.
		if within(pass, row.pkgpath) {
			continue
		}

		var (
			lenMethod = index.Selection(row.pkgpath, row.typename, row.lenmethod)
			atMethod  = index.Selection(row.pkgpath, row.typename, row.atmethod)
		)

		// chooseName returns an appropriate fresh name
		// for the index variable of the iterator loop
		// whose body is specified.
		//
		// If the loop body starts with
		//
		//     for ... { e := x.At(i); use(e) }
		//
		// or
		//
		//     for ... { if e := x.At(i); cond { use(e) } }
		//
		// then chooseName prefers the name e and additionally
		// returns the var's symbol. We'll transform this to:
		//
		//     for e := range x.Len() { e := e; use(e) }
		//
		// which leaves a redundant assignment that a
		// subsequent 'forvar' pass will eliminate.
		chooseName := func(curBody inspector.Cursor, x ast.Expr, i *types.Var) (string, *types.Var) {

			// isVarAssign reports whether stmt has the form v := x.At(i)
			// and returns the variable if so.
			isVarAssign := func(stmt ast.Stmt) *types.Var {
				if assign, ok := stmt.(*ast.AssignStmt); ok &&
					assign.Tok == token.DEFINE &&
					len(assign.Lhs) == 1 &&
					len(assign.Rhs) == 1 &&
					is[*ast.Ident](assign.Lhs[0]) {
					// call to x.At(i)?
					if call, ok := assign.Rhs[0].(*ast.CallExpr); ok &&
						typeutil.Callee(info, call) == atMethod &&
						astutil.EqualSyntax(ast.Unparen(call.Fun).(*ast.SelectorExpr).X, x) &&
						is[*ast.Ident](call.Args[0]) &&
						info.Uses[call.Args[0].(*ast.Ident)] == i {
						// Have: elem := x.At(i)
						id := assign.Lhs[0].(*ast.Ident)
						return info.Defs[id].(*types.Var)
					}
				}
				return nil
			}

			body := curBody.Node().(*ast.BlockStmt)
			if len(body.List) > 0 {
				// Is body { elem := x.At(i); ... } ?
				if v := isVarAssign(body.List[0]); v != nil {
					return v.Name(), v
				}

				// Or { if elem := x.At(i); cond { ... } } ?
				if ifstmt, ok := body.List[0].(*ast.IfStmt); ok && ifstmt.Init != nil {
					if v := isVarAssign(ifstmt.Init); v != nil {
						return v.Name(), v
					}
				}
			}

			loop := curBody.Parent().Node()

			// Choose a fresh name only if
			// (a) the preferred name is already declared here, and
			// (b) there are references to it from the loop body.
			// TODO(adonovan): this pattern also appears in errorsastype,
			// and is wanted elsewhere; factor.
			name := row.elemname
			if v := lookup(info, curBody, name); v != nil {
				// is it free in body?
				for curUse := range index.Uses(v) {
					if curBody.Contains(curUse) {
						name = refactor.FreshName(info.Scopes[loop], loop.Pos(), name)
						break
					}
				}
			}
			return name, nil
		}

		// Process each call of x.Len().
	nextCall:
		for curLenCall := range index.Calls(lenMethod) {
			lenSel, ok := ast.Unparen(curLenCall.Node().(*ast.CallExpr).Fun).(*ast.SelectorExpr)
			if !ok {
				continue
			}
			// lenSel is "x.Len"

			var (
				rng      analysis.Range   // where to report diagnostic
				curBody  inspector.Cursor // loop body
				indexVar *types.Var       // old loop index var
				elemVar  *types.Var       // existing "elem := x.At(i)" var, if present
				elem     string           // name for new loop var
				edits    []analysis.TextEdit
			)

			// Analyze enclosing loop.
			switch ek, _ := curLenCall.ParentEdge(); ek {
			case edge.BinaryExpr_Y:
				// pattern 1: for i := 0; i < x.Len(); i++ { ... }
				var (
					curCmp = curLenCall.Parent()
					cmp    = curCmp.Node().(*ast.BinaryExpr)
				)
				if cmp.Op != token.LSS ||
					!astutil.IsChildOf(curCmp, edge.ForStmt_Cond) {
					continue
				}
				if id, ok := cmp.X.(*ast.Ident); ok {
					// Have: for _; i < x.Len(); _ { ... }
					var (
						v      = info.Uses[id].(*types.Var)
						curFor = curCmp.Parent()
						loop   = curFor.Node().(*ast.ForStmt)
					)
					if v != isIncrementLoop(info, loop) {
						continue
					}
					// Have: for i := 0; i < x.Len(); i++ { ... }.
					//       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
					rng = astutil.RangeOf(loop.For, loop.Post.End())
					indexVar = v
					curBody = curFor.ChildAt(edge.ForStmt_Body, -1)
					elem, elemVar = chooseName(curBody, lenSel.X, indexVar)

					//	for i    := 0; i < x.Len(); i++ {
					//          ----    -------  ---  -----
					//	for elem := range  x.All()      {
					edits = []analysis.TextEdit{
						{
							Pos:     v.Pos(),
							End:     v.Pos() + token.Pos(len(v.Name())),
							NewText: []byte(elem),
						},
						{
							Pos:     loop.Init.(*ast.AssignStmt).Rhs[0].Pos(),
							End:     cmp.Y.Pos(),
							NewText: []byte("range "),
						},
						{
							Pos:     lenSel.Sel.Pos(),
							End:     lenSel.Sel.End(),
							NewText: []byte(row.itermethod),
						},
						{
							Pos: curLenCall.Node().End(),
							End: loop.Post.End(),
						},
					}
				}

			case edge.RangeStmt_X:
				// pattern 2: for i := range x.Len() { ... }
				var (
					curRange = curLenCall.Parent()
					loop     = curRange.Node().(*ast.RangeStmt)
				)
				if id, ok := loop.Key.(*ast.Ident); ok &&
					loop.Value == nil &&
					loop.Tok == token.DEFINE {
					// Have: for i := range x.Len() { ... }
					//                ~~~~~~~~~~~~~

					rng = astutil.RangeOf(loop.Range, loop.X.End())
					indexVar = info.Defs[id].(*types.Var)
					curBody = curRange.ChildAt(edge.RangeStmt_Body, -1)
					elem, elemVar = chooseName(curBody, lenSel.X, indexVar)

					//	for i    := range x.Len() {
					//          ----            ---
					//	for elem := range x.All() {
					edits = []analysis.TextEdit{
						{
							Pos:     loop.Key.Pos(),
							End:     loop.Key.End(),
							NewText: []byte(elem),
						},
						{
							Pos:     lenSel.Sel.Pos(),
							End:     lenSel.Sel.End(),
							NewText: []byte(row.itermethod),
						},
					}
				}
			}

			if indexVar == nil {
				continue // no loop of the required form
			}

			// TODO(adonovan): what about possible
			// modifications of x within the loop?
			// Aliasing seems to make a conservative
			// treatment impossible.

			// Check that all uses of var i within loop body are x.At(i).
			for curUse := range index.Uses(indexVar) {
				if !curBody.Contains(curUse) {
					continue
				}
				if ek, argidx := curUse.ParentEdge(); ek != edge.CallExpr_Args || argidx != 0 {
					continue nextCall // use is not arg of call
				}
				curAtCall := curUse.Parent()
				atCall := curAtCall.Node().(*ast.CallExpr)
				if typeutil.Callee(info, atCall) != atMethod {
					continue nextCall // use is not arg of call to T.At
				}
				atSel := ast.Unparen(atCall.Fun).(*ast.SelectorExpr)

				// Check receivers of Len, At calls match (syntactically).
				if !astutil.EqualSyntax(lenSel.X, atSel.X) {
					continue nextCall
				}

				// At each point of use, check that
				// the fresh variable is not shadowed
				// by an intervening local declaration
				// (or by the idiomatic elemVar optionally
				// found by chooseName).
				if obj := lookup(info, curAtCall, elem); obj != nil && obj != elemVar && obj.Pos() > indexVar.Pos() {
					// (Ideally, instead of giving up, we would
					// embellish the name and try again.)
					continue nextCall
				}

				// use(x.At(i))
				//     -------
				// use(elem   )
				edits = append(edits, analysis.TextEdit{
					Pos:     atCall.Pos(),
					End:     atCall.End(),
					NewText: []byte(elem),
				})
			}

			// Check file Go version is new enough for the iterator method.
			// (In the long run, version filters are not highly selective,
			// so there's no need to do them first, especially as this check
			// may be somewhat expensive.)
			if v, ok := methodGoVersion(row.pkgpath, row.typename, row.itermethod); !ok {
				panic("no version found")
			} else if !analyzerutil.FileUsesGoVersion(pass, astutil.EnclosingFile(curLenCall), v.String()) {
				continue nextCall
			}

			pass.Report(analysis.Diagnostic{
				Pos: rng.Pos(),
				End: rng.End(),
				Message: fmt.Sprintf("%s/%s loop can simplified using %s.%s iteration",
					row.lenmethod, row.atmethod, row.typename, row.itermethod),
				SuggestedFixes: []analysis.SuggestedFix{{
					Message: fmt.Sprintf(
						"Replace %s/%s loop with %s.%s iteration",
						row.lenmethod, row.atmethod, row.typename, row.itermethod),
					TextEdits: edits,
				}},
			})
		}
	}
	return nil, nil
}

// -- helpers --

// methodGoVersion reports the version at which the method
// (pkgpath.recvtype).method appeared in the standard library.
func methodGoVersion(pkgpath, recvtype, method string) (stdlib.Version, bool) {
	// TODO(adonovan): opt: this might be inefficient for large packages
	// like go/types. If so, memoize using a map (and kill two birds with
	// one stone by also memoizing the 'within' check above).
	for _, sym := range stdlib.PackageSymbols[pkgpath] {
		if sym.Kind == stdlib.Method {
			_, recv, name := sym.SplitMethod()
			if recv == recvtype && name == method {
				return sym.Version, true
			}
		}
	}
	return 0, false
}
