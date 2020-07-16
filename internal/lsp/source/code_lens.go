// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"go/ast"
	"go/token"
	"go/types"
	"path/filepath"
	"regexp"
	"strings"

	"golang.org/x/tools/internal/lsp/protocol"
)

type lensFunc func(context.Context, Snapshot, FileHandle) ([]protocol.CodeLens, error)

var lensFuncs = map[string]lensFunc{
	CommandGenerate:      goGenerateCodeLens,
	CommandTest:          runTestCodeLens,
	CommandRegenerateCgo: regenerateCgoLens,
}

// CodeLens computes code lens for Go source code.
func CodeLens(ctx context.Context, snapshot Snapshot, fh FileHandle) ([]protocol.CodeLens, error) {
	var result []protocol.CodeLens
	for lens, lf := range lensFuncs {
		if !snapshot.View().Options().EnabledCodeLens[lens] {
			continue
		}
		added, err := lf(ctx, snapshot, fh)
		if err != nil {
			return nil, err
		}
		result = append(result, added...)
	}
	return result, nil
}

func runTestCodeLens(ctx context.Context, snapshot Snapshot, fh FileHandle) ([]protocol.CodeLens, error) {
	codeLens := make([]protocol.CodeLens, 0)

	if !strings.HasSuffix(fh.URI().Filename(), "_test.go") {
		return nil, nil
	}
	pkg, pgh, err := getParsedFile(ctx, snapshot, fh, WidestPackageHandle)
	if err != nil {
		return nil, err
	}
	file, _, m, _, err := pgh.Cached()
	if err != nil {
		return nil, err
	}
	for _, d := range file.Decls {
		fn, ok := d.(*ast.FuncDecl)
		if !ok {
			continue
		}
		if !isTestFunc(fn, pkg) {
			continue
		}
		fset := snapshot.View().Session().Cache().FileSet()
		rng, err := newMappedRange(fset, m, d.Pos(), d.Pos()).Range()
		if err != nil {
			return nil, err
		}
		codeLens = append(codeLens, protocol.CodeLens{
			Range: rng,
			Command: protocol.Command{
				Title:     "run test",
				Command:   CommandTest,
				Arguments: []interface{}{fn.Name.Name, fh.URI()},
			},
		})
	}
	return codeLens, nil
}

var (
	testRe      = regexp.MustCompile("^Test[^a-z]")
	benchmarkRe = regexp.MustCompile("^Benchmark[^a-z]")
)

func isTestFunc(fn *ast.FuncDecl, pkg Package) bool {
	// Make sure that the function name matches either a test or benchmark function.
	if !(testRe.MatchString(fn.Name.Name) || benchmarkRe.MatchString(fn.Name.Name)) {
		return false
	}
	info := pkg.GetTypesInfo()
	if info == nil {
		return false
	}
	obj := info.ObjectOf(fn.Name)
	if obj == nil {
		return false
	}
	sig, ok := obj.Type().(*types.Signature)
	if !ok {
		return false
	}
	// Test functions should have only one parameter.
	if sig.Params().Len() != 1 {
		return false
	}

	// Check the type of the only parameter to confirm that it is *testing.T
	// or *testing.B.
	paramTyp, ok := sig.Params().At(0).Type().(*types.Pointer)
	if !ok {
		return false
	}
	named, ok := paramTyp.Elem().(*types.Named)
	if !ok {
		return false
	}
	namedObj := named.Obj()
	if namedObj.Pkg().Path() != "testing" {
		return false
	}
	paramName := namedObj.Id()
	return paramName == "T" || paramName == "B"
}

func goGenerateCodeLens(ctx context.Context, snapshot Snapshot, fh FileHandle) ([]protocol.CodeLens, error) {
	pgh := snapshot.View().Session().Cache().ParseGoHandle(ctx, fh, ParseFull)
	file, _, m, _, err := pgh.Parse(ctx)
	if err != nil {
		return nil, err
	}
	const ggDirective = "//go:generate"
	for _, c := range file.Comments {
		for _, l := range c.List {
			if !strings.HasPrefix(l.Text, ggDirective) {
				continue
			}
			fset := snapshot.View().Session().Cache().FileSet()
			rng, err := newMappedRange(fset, m, l.Pos(), l.Pos()+token.Pos(len(ggDirective))).Range()
			if err != nil {
				return nil, err
			}
			dir := filepath.Dir(fh.URI().Filename())
			return []protocol.CodeLens{
				{
					Range: rng,
					Command: protocol.Command{
						Title:     "run go generate",
						Command:   CommandGenerate,
						Arguments: []interface{}{dir, false},
					},
				},
				{
					Range: rng,
					Command: protocol.Command{
						Title:     "run go generate ./...",
						Command:   CommandGenerate,
						Arguments: []interface{}{dir, true},
					},
				},
			}, nil

		}
	}
	return nil, nil
}

func regenerateCgoLens(ctx context.Context, snapshot Snapshot, fh FileHandle) ([]protocol.CodeLens, error) {
	pgh := snapshot.View().Session().Cache().ParseGoHandle(ctx, fh, ParseFull)
	file, _, m, _, err := pgh.Parse(ctx)
	if err != nil {
		return nil, err
	}
	var c *ast.ImportSpec
	for _, imp := range file.Imports {
		if imp.Path.Value == `"C"` {
			c = imp
		}
	}
	if c == nil {
		return nil, nil
	}
	fset := snapshot.View().Session().Cache().FileSet()
	rng, err := newMappedRange(fset, m, c.Pos(), c.EndPos).Range()
	if err != nil {
		return nil, err
	}
	return []protocol.CodeLens{
		{
			Range: rng,
			Command: protocol.Command{
				Title:     "regenerate cgo definitions",
				Command:   CommandRegenerateCgo,
				Arguments: []interface{}{fh.URI()},
			},
		},
	}, nil
}
