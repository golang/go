// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.20

package analyzer

import (
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"os"
	"strings"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/go/types/typeutil"
	"golang.org/x/tools/internal/diff"
	"golang.org/x/tools/internal/refactor/inline"
)

const Doc = `inline calls to functions with "inlineme" doc comment`

var Analyzer = &analysis.Analyzer{
	Name:      "inline",
	Doc:       Doc,
	URL:       "https://pkg.go.dev/golang.org/x/tools/internal/refactor/inline/analyzer",
	Run:       run,
	FactTypes: []analysis.Fact{new(inlineMeFact)},
	Requires:  []*analysis.Analyzer{inspect.Analyzer},
}

func run(pass *analysis.Pass) (interface{}, error) {
	// Memoize repeated calls for same file.
	// TODO(adonovan): the analysis.Pass should abstract this (#62292)
	// as the driver may not be reading directly from the file system.
	fileContent := make(map[string][]byte)
	readFile := func(node ast.Node) ([]byte, error) {
		filename := pass.Fset.File(node.Pos()).Name()
		content, ok := fileContent[filename]
		if !ok {
			var err error
			content, err = os.ReadFile(filename)
			if err != nil {
				return nil, err
			}
			fileContent[filename] = content
		}
		return content, nil
	}

	// Pass 1: find functions annotated with an "inlineme"
	// comment, and export a fact for each one.
	inlinable := make(map[*types.Func]*inline.Callee) // memoization of fact import (nil => no fact)
	for _, file := range pass.Files {
		for _, decl := range file.Decls {
			if decl, ok := decl.(*ast.FuncDecl); ok {
				// TODO(adonovan): this is just a placeholder.
				// Use the precise go:fix syntax in the proposal.
				// Beware that //go: comments are treated specially
				// by (*ast.CommentGroup).Text().
				// TODO(adonovan): alternatively, consider using
				// the universal annotation mechanism sketched in
				// https://go.dev/cl/489835 (which doesn't yet have
				// a proper proposal).
				if strings.Contains(decl.Doc.Text(), "inlineme") {
					content, err := readFile(file)
					if err != nil {
						pass.Reportf(decl.Doc.Pos(), "invalid inlining candidate: cannot read source file: %v", err)
						continue
					}
					callee, err := inline.AnalyzeCallee(discard, pass.Fset, pass.Pkg, pass.TypesInfo, decl, content)
					if err != nil {
						pass.Reportf(decl.Doc.Pos(), "invalid inlining candidate: %v", err)
						continue
					}
					fn := pass.TypesInfo.Defs[decl.Name].(*types.Func)
					pass.ExportObjectFact(fn, &inlineMeFact{callee})
					inlinable[fn] = callee
				}
			}
		}
	}

	// Pass 2. Inline each static call to an inlinable function.
	inspect := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)
	nodeFilter := []ast.Node{
		(*ast.File)(nil),
		(*ast.CallExpr)(nil),
	}
	var currentFile *ast.File
	inspect.Preorder(nodeFilter, func(n ast.Node) {
		if file, ok := n.(*ast.File); ok {
			currentFile = file
			return
		}
		call := n.(*ast.CallExpr)
		if fn := typeutil.StaticCallee(pass.TypesInfo, call); fn != nil {
			// Inlinable?
			callee, ok := inlinable[fn]
			if !ok {
				var fact inlineMeFact
				if pass.ImportObjectFact(fn, &fact) {
					callee = fact.callee
					inlinable[fn] = callee
				}
			}
			if callee == nil {
				return // nope
			}

			// Inline the call.
			content, err := readFile(call)
			if err != nil {
				pass.Reportf(call.Lparen, "invalid inlining candidate: cannot read source file: %v", err)
				return
			}
			caller := &inline.Caller{
				Fset:    pass.Fset,
				Types:   pass.Pkg,
				Info:    pass.TypesInfo,
				File:    currentFile,
				Call:    call,
				Content: content,
			}
			got, err := inline.Inline(discard, caller, callee)
			if err != nil {
				pass.Reportf(call.Lparen, "%v", err)
				return
			}

			// Suggest the "fix".
			var textEdits []analysis.TextEdit
			for _, edit := range diff.Bytes(content, got) {
				textEdits = append(textEdits, analysis.TextEdit{
					Pos:     currentFile.FileStart + token.Pos(edit.Start),
					End:     currentFile.FileStart + token.Pos(edit.End),
					NewText: []byte(edit.New),
				})
			}
			msg := fmt.Sprintf("inline call of %v", callee)
			pass.Report(analysis.Diagnostic{
				Pos:     call.Pos(),
				End:     call.End(),
				Message: msg,
				SuggestedFixes: []analysis.SuggestedFix{{
					Message:   msg,
					TextEdits: textEdits,
				}},
			})
		}
	})

	return nil, nil
}

type inlineMeFact struct{ callee *inline.Callee }

func (f *inlineMeFact) String() string { return "inlineme " + f.callee.String() }
func (*inlineMeFact) AFact()           {}

func discard(string, ...any) {}
