// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modernize

import (
	"fmt"
	"go/ast"
	"go/token"
	"strings"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/types/typeutil"
	"golang.org/x/tools/internal/analysis/analyzerutil"
	typeindexanalyzer "golang.org/x/tools/internal/analysis/typeindex"
	"golang.org/x/tools/internal/astutil"
	"golang.org/x/tools/internal/refactor"
	"golang.org/x/tools/internal/typesinternal"
	"golang.org/x/tools/internal/typesinternal/typeindex"
	"golang.org/x/tools/internal/versions"
)

var StringsCutPrefixAnalyzer = &analysis.Analyzer{
	Name: "stringscutprefix",
	Doc:  analyzerutil.MustExtractDoc(doc, "stringscutprefix"),
	Requires: []*analysis.Analyzer{
		inspect.Analyzer,
		typeindexanalyzer.Analyzer,
	},
	Run: stringscutprefix,
	URL: "https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/modernize#stringscutprefix",
}

// stringscutprefix offers a fix to replace an if statement which
// calls to the 2 patterns below with strings.CutPrefix or strings.CutSuffix.
//
// Patterns:
//
//  1. if strings.HasPrefix(s, pre) { use(strings.TrimPrefix(s, pre) }
//     =>
//     if after, ok := strings.CutPrefix(s, pre); ok { use(after) }
//
//  2. if after := strings.TrimPrefix(s, pre); after != s { use(after) }
//     =>
//     if after, ok := strings.CutPrefix(s, pre); ok { use(after) }
//
// Similar patterns apply for CutSuffix.
//
// The use must occur within the first statement of the block, and the offered fix
// only replaces the first occurrence of strings.TrimPrefix/TrimSuffix.
//
// Variants:
// - bytes.HasPrefix/HasSuffix usage as pattern 1.
func stringscutprefix(pass *analysis.Pass) (any, error) {
	var (
		index = pass.ResultOf[typeindexanalyzer.Analyzer].(*typeindex.Index)
		info  = pass.TypesInfo

		stringsTrimPrefix = index.Object("strings", "TrimPrefix")
		bytesTrimPrefix   = index.Object("bytes", "TrimPrefix")
		stringsTrimSuffix = index.Object("strings", "TrimSuffix")
		bytesTrimSuffix   = index.Object("bytes", "TrimSuffix")
	)
	if !index.Used(stringsTrimPrefix, bytesTrimPrefix, stringsTrimSuffix, bytesTrimSuffix) {
		return nil, nil
	}

	for curFile := range filesUsingGoVersion(pass, versions.Go1_20) {
		for curIfStmt := range curFile.Preorder((*ast.IfStmt)(nil)) {
			ifStmt := curIfStmt.Node().(*ast.IfStmt)

			// pattern1
			if call, ok := ifStmt.Cond.(*ast.CallExpr); ok && ifStmt.Init == nil && len(ifStmt.Body.List) > 0 {

				obj := typeutil.Callee(info, call)
				if !typesinternal.IsFunctionNamed(obj, "strings", "HasPrefix", "HasSuffix") &&
					!typesinternal.IsFunctionNamed(obj, "bytes", "HasPrefix", "HasSuffix") {
					continue
				}
				isPrefix := strings.HasSuffix(obj.Name(), "Prefix")

				// Replace the first occurrence of strings.TrimPrefix(s, pre) in the first statement only,
				// but not later statements in case s or pre are modified by intervening logic (ditto Suffix).
				firstStmt := curIfStmt.Child(ifStmt.Body).Child(ifStmt.Body.List[0])
				for curCall := range firstStmt.Preorder((*ast.CallExpr)(nil)) {
					call1 := curCall.Node().(*ast.CallExpr)
					obj1 := typeutil.Callee(info, call1)
					// bytesTrimPrefix or stringsTrimPrefix might be nil if the file doesn't import it,
					// so we need to ensure the obj1 is not nil otherwise the call1 is not TrimPrefix and cause a panic (ditto Suffix).
					if obj1 == nil ||
						obj1 != stringsTrimPrefix && obj1 != bytesTrimPrefix &&
							obj1 != stringsTrimSuffix && obj1 != bytesTrimSuffix {
						continue
					}

					isPrefix1 := strings.HasSuffix(obj1.Name(), "Prefix")
					var cutFuncName, varName, message, fixMessage string
					if isPrefix && isPrefix1 {
						cutFuncName = "CutPrefix"
						varName = "after"
						message = "HasPrefix + TrimPrefix can be simplified to CutPrefix"
						fixMessage = "Replace HasPrefix/TrimPrefix with CutPrefix"
					} else if !isPrefix && !isPrefix1 {
						cutFuncName = "CutSuffix"
						varName = "before"
						message = "HasSuffix + TrimSuffix can be simplified to CutSuffix"
						fixMessage = "Replace HasSuffix/TrimSuffix with CutSuffix"
					} else {
						continue
					}

					// Have: if strings.HasPrefix(s0, pre0) { ...strings.TrimPrefix(s, pre)... } (ditto Suffix)
					var (
						s0   = call.Args[0]
						pre0 = call.Args[1]
						s    = call1.Args[0]
						pre  = call1.Args[1]
					)

					// check whether the obj1 uses the exact the same argument with strings.HasPrefix
					// shadow variables won't be valid because we only access the first statement (ditto Suffix).
					if astutil.EqualSyntax(s0, s) && astutil.EqualSyntax(pre0, pre) {
						after := refactor.FreshName(info.Scopes[ifStmt], ifStmt.Pos(), varName)
						prefix, importEdits := refactor.AddImport(
							info,
							curFile.Node().(*ast.File),
							obj1.Pkg().Name(),
							obj1.Pkg().Path(),
							cutFuncName,
							call.Pos(),
						)
						okVarName := refactor.FreshName(info.Scopes[ifStmt], ifStmt.Pos(), "ok")
						pass.Report(analysis.Diagnostic{
							// highlight at HasPrefix call (ditto Suffix).
							Pos:     call.Pos(),
							End:     call.End(),
							Message: message,
							SuggestedFixes: []analysis.SuggestedFix{{
								Message: fixMessage,
								// if              strings.HasPrefix(s, pre)     { use(strings.TrimPrefix(s, pre)) }
								//    ------------ -----------------        -----      --------------------------
								// if after, ok := strings.CutPrefix(s, pre); ok { use(after)                      }
								// (ditto Suffix)
								TextEdits: append(importEdits, []analysis.TextEdit{
									{
										Pos:     call.Fun.Pos(),
										End:     call.Fun.Pos(),
										NewText: fmt.Appendf(nil, "%s, %s :=", after, okVarName),
									},
									{
										Pos:     call.Fun.Pos(),
										End:     call.Fun.End(),
										NewText: fmt.Appendf(nil, "%s%s", prefix, cutFuncName),
									},
									{
										Pos:     call.End(),
										End:     call.End(),
										NewText: fmt.Appendf(nil, "; %s ", okVarName),
									},
									{
										Pos:     call1.Pos(),
										End:     call1.End(),
										NewText: []byte(after),
									},
								}...),
							}}},
						)
						break
					}
				}
			}

			// pattern2
			if bin, ok := ifStmt.Cond.(*ast.BinaryExpr); ok &&
				bin.Op == token.NEQ &&
				ifStmt.Init != nil &&
				isSimpleAssign(ifStmt.Init) {
				assign := ifStmt.Init.(*ast.AssignStmt)
				if call, ok := assign.Rhs[0].(*ast.CallExpr); ok && assign.Tok == token.DEFINE {
					lhs := assign.Lhs[0]
					obj := typeutil.Callee(info, call)

					if obj == nil ||
						obj != stringsTrimPrefix && obj != bytesTrimPrefix && obj != stringsTrimSuffix && obj != bytesTrimSuffix {
						continue
					}

					isPrefix1 := strings.HasSuffix(obj.Name(), "Prefix")
					var cutFuncName, message, fixMessage string
					if isPrefix1 {
						cutFuncName = "CutPrefix"
						message = "TrimPrefix can be simplified to CutPrefix"
						fixMessage = "Replace TrimPrefix with CutPrefix"
					} else {
						cutFuncName = "CutSuffix"
						message = "TrimSuffix can be simplified to CutSuffix"
						fixMessage = "Replace TrimSuffix with CutSuffix"
					}

					if astutil.EqualSyntax(lhs, bin.X) && astutil.EqualSyntax(call.Args[0], bin.Y) ||
						(astutil.EqualSyntax(lhs, bin.Y) && astutil.EqualSyntax(call.Args[0], bin.X)) {
						// TODO(adonovan): avoid FreshName when not needed; see errorsastype.
						okVarName := refactor.FreshName(info.Scopes[ifStmt], ifStmt.Pos(), "ok")
						// Have one of:
						//   if rest := TrimPrefix(s, prefix); rest != s { (ditto Suffix)
						//   if rest := TrimPrefix(s, prefix); s != rest { (ditto Suffix)

						// We use AddImport not to add an import (since it exists already)
						// but to compute the correct prefix in the dot-import case.
						prefix, importEdits := refactor.AddImport(
							info,
							curFile.Node().(*ast.File),
							obj.Pkg().Name(),
							obj.Pkg().Path(),
							cutFuncName,
							call.Pos(),
						)

						pass.Report(analysis.Diagnostic{
							// highlight from the init and the condition end.
							Pos:     ifStmt.Init.Pos(),
							End:     ifStmt.Cond.End(),
							Message: message,
							SuggestedFixes: []analysis.SuggestedFix{{
								Message: fixMessage,
								// if x     := strings.TrimPrefix(s, pre); x != s ...
								//     ----            ----------          ------
								// if x, ok := strings.CutPrefix (s, pre); ok     ...
								// (ditto Suffix)
								TextEdits: append(importEdits, []analysis.TextEdit{
									{
										Pos:     assign.Lhs[0].End(),
										End:     assign.Lhs[0].End(),
										NewText: fmt.Appendf(nil, ", %s", okVarName),
									},
									{
										Pos:     call.Fun.Pos(),
										End:     call.Fun.End(),
										NewText: fmt.Appendf(nil, "%s%s", prefix, cutFuncName),
									},
									{
										Pos:     ifStmt.Cond.Pos(),
										End:     ifStmt.Cond.End(),
										NewText: []byte(okVarName),
									},
								}...),
							}},
						})
					}
				}
			}
		}
	}
	return nil, nil
}
