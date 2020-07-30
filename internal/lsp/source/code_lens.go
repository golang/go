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
	"golang.org/x/tools/internal/span"
)

type lensFunc func(context.Context, Snapshot, FileHandle) ([]protocol.CodeLens, error)

var lensFuncs = map[string]lensFunc{
	CommandGenerate.Name:      goGenerateCodeLens,
	CommandTest.Name:          runTestCodeLens,
	CommandRegenerateCgo.Name: regenerateCgoLens,
	CommandToggleDetails.Name: toggleDetailsCodeLens,
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

var (
	testRe      = regexp.MustCompile("^Test[^a-z]")
	benchmarkRe = regexp.MustCompile("^Benchmark[^a-z]")
)

func runTestCodeLens(ctx context.Context, snapshot Snapshot, fh FileHandle) ([]protocol.CodeLens, error) {
	codeLens := make([]protocol.CodeLens, 0)

	if !strings.HasSuffix(fh.URI().Filename(), "_test.go") {
		return nil, nil
	}
	pkg, pgf, err := getParsedFile(ctx, snapshot, fh, WidestPackage)
	if err != nil {
		return nil, err
	}

	var benchFns []string
	for _, d := range pgf.File.Decls {
		fn, ok := d.(*ast.FuncDecl)
		if !ok {
			continue
		}
		if benchmarkRe.MatchString(fn.Name.Name) {
			benchFns = append(benchFns, fn.Name.Name)
		}
		rng, err := newMappedRange(snapshot.FileSet(), pgf.Mapper, d.Pos(), d.Pos()).Range()
		if err != nil {
			return nil, err
		}

		if matchTestFunc(fn, pkg, testRe, "T") {
			jsonArgs, err := MarshalArgs(fh.URI(), []string{fn.Name.Name}, nil)
			if err != nil {
				return nil, err
			}
			codeLens = append(codeLens, protocol.CodeLens{
				Range: rng,
				Command: protocol.Command{
					Title:     "run test",
					Command:   CommandTest.Name,
					Arguments: jsonArgs,
				},
			})
		}

		if matchTestFunc(fn, pkg, benchmarkRe, "B") {
			jsonArgs, err := MarshalArgs(fh.URI(), nil, []string{fn.Name.Name})
			if err != nil {
				return nil, err
			}
			codeLens = append(codeLens, protocol.CodeLens{
				Range: rng,
				Command: protocol.Command{
					Title:     "run benchmark",
					Command:   CommandTest.Name,
					Arguments: jsonArgs,
				},
			})
		}
	}
	// add a code lens to the top of the file which runs all benchmarks in the file
	rng, err := newMappedRange(snapshot.FileSet(), pgf.Mapper, pgf.File.Package, pgf.File.Package).Range()
	if err != nil {
		return nil, err
	}
	args, err := MarshalArgs(fh.URI(), []string{}, benchFns)
	if err != nil {
		return nil, err
	}
	codeLens = append(codeLens, protocol.CodeLens{
		Range: rng,
		Command: protocol.Command{
			Title:     "run file benchmarks",
			Command:   CommandTest.Name,
			Arguments: args,
		},
	})
	return codeLens, nil
}

func matchTestFunc(fn *ast.FuncDecl, pkg Package, nameRe *regexp.Regexp, paramID string) bool {
	// Make sure that the function name matches a test function.
	if !nameRe.MatchString(fn.Name.Name) {
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

	// Check the type of the only parameter
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
	return namedObj.Id() == paramID
}

func goGenerateCodeLens(ctx context.Context, snapshot Snapshot, fh FileHandle) ([]protocol.CodeLens, error) {
	pgf, err := snapshot.ParseGo(ctx, fh, ParseFull)
	if err != nil {
		return nil, err
	}
	const ggDirective = "//go:generate"
	for _, c := range pgf.File.Comments {
		for _, l := range c.List {
			if !strings.HasPrefix(l.Text, ggDirective) {
				continue
			}
			rng, err := newMappedRange(snapshot.FileSet(), pgf.Mapper, l.Pos(), l.Pos()+token.Pos(len(ggDirective))).Range()
			if err != nil {
				return nil, err
			}
			dir := span.URIFromPath(filepath.Dir(fh.URI().Filename()))
			nonRecursiveArgs, err := MarshalArgs(dir, false)
			if err != nil {
				return nil, err
			}
			recursiveArgs, err := MarshalArgs(dir, true)
			if err != nil {
				return nil, err
			}
			return []protocol.CodeLens{
				{
					Range: rng,
					Command: protocol.Command{
						Title:     "run go generate",
						Command:   CommandGenerate.Name,
						Arguments: nonRecursiveArgs,
					},
				},
				{
					Range: rng,
					Command: protocol.Command{
						Title:     "run go generate ./...",
						Command:   CommandGenerate.Name,
						Arguments: recursiveArgs,
					},
				},
			}, nil

		}
	}
	return nil, nil
}

func regenerateCgoLens(ctx context.Context, snapshot Snapshot, fh FileHandle) ([]protocol.CodeLens, error) {
	pgf, err := snapshot.ParseGo(ctx, fh, ParseFull)
	if err != nil {
		return nil, err
	}
	var c *ast.ImportSpec
	for _, imp := range pgf.File.Imports {
		if imp.Path.Value == `"C"` {
			c = imp
		}
	}
	if c == nil {
		return nil, nil
	}
	rng, err := newMappedRange(snapshot.FileSet(), pgf.Mapper, c.Pos(), c.EndPos).Range()
	if err != nil {
		return nil, err
	}
	jsonArgs, err := MarshalArgs(fh.URI())
	if err != nil {
		return nil, err
	}
	return []protocol.CodeLens{
		{
			Range: rng,
			Command: protocol.Command{
				Title:     "regenerate cgo definitions",
				Command:   CommandRegenerateCgo.Name,
				Arguments: jsonArgs,
			},
		},
	}, nil
}

func toggleDetailsCodeLens(ctx context.Context, snapshot Snapshot, fh FileHandle) ([]protocol.CodeLens, error) {
	_, pgf, err := getParsedFile(ctx, snapshot, fh, WidestPackage)
	if err != nil {
		return nil, err
	}
	rng, err := newMappedRange(snapshot.FileSet(), pgf.Mapper, pgf.File.Package, pgf.File.Package).Range()
	if err != nil {
		return nil, err
	}
	jsonArgs, err := MarshalArgs(fh.URI())
	if err != nil {
		return nil, err
	}
	return []protocol.CodeLens{
		{
			Range: rng,
			Command: protocol.Command{
				Title:     "Toggle gc annotation details",
				Command:   CommandToggleDetails.Name,
				Arguments: jsonArgs,
			},
		},
	}, nil
}
