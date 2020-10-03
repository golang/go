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

type LensFunc func(context.Context, Snapshot, FileHandle) ([]protocol.CodeLens, error)

// LensFuncs returns the supported lensFuncs for Go files.
func LensFuncs() map[string]LensFunc {
	return map[string]LensFunc{
		CommandGenerate.Name:      goGenerateCodeLens,
		CommandTest.Name:          runTestCodeLens,
		CommandRegenerateCgo.Name: regenerateCgoLens,
		CommandToggleDetails.Name: toggleDetailsCodeLens,
	}
}

var (
	testRe      = regexp.MustCompile("^Test[^a-z]")
	benchmarkRe = regexp.MustCompile("^Benchmark[^a-z]")
)

func runTestCodeLens(ctx context.Context, snapshot Snapshot, fh FileHandle) ([]protocol.CodeLens, error) {
	codeLens := make([]protocol.CodeLens, 0)

	fns, err := TestsAndBenchmarks(ctx, snapshot, fh)
	if err != nil {
		return nil, err
	}
	for _, fn := range fns.Tests {
		jsonArgs, err := MarshalArgs(fh.URI(), []string{fn.Name}, nil)
		if err != nil {
			return nil, err
		}
		codeLens = append(codeLens, protocol.CodeLens{
			Range: protocol.Range{Start: fn.Rng.Start, End: fn.Rng.Start},
			Command: protocol.Command{
				Title:     "run test",
				Command:   CommandTest.ID(),
				Arguments: jsonArgs,
			},
		})
	}

	for _, fn := range fns.Benchmarks {
		jsonArgs, err := MarshalArgs(fh.URI(), nil, []string{fn.Name})
		if err != nil {
			return nil, err
		}
		codeLens = append(codeLens, protocol.CodeLens{
			Range: protocol.Range{Start: fn.Rng.Start, End: fn.Rng.Start},
			Command: protocol.Command{
				Title:     "run benchmark",
				Command:   CommandTest.ID(),
				Arguments: jsonArgs,
			},
		})
	}

	_, pgf, err := GetParsedFile(ctx, snapshot, fh, WidestPackage)
	if err != nil {
		return nil, err
	}
	// add a code lens to the top of the file which runs all benchmarks in the file
	rng, err := NewMappedRange(snapshot.FileSet(), pgf.Mapper, pgf.File.Package, pgf.File.Package).Range()
	if err != nil {
		return nil, err
	}
	args, err := MarshalArgs(fh.URI(), []string{}, fns.Benchmarks)
	if err != nil {
		return nil, err
	}
	codeLens = append(codeLens, protocol.CodeLens{
		Range: rng,
		Command: protocol.Command{
			Title:     "run file benchmarks",
			Command:   CommandTest.ID(),
			Arguments: args,
		},
	})
	return codeLens, nil
}

type testFn struct {
	Name string
	Rng  protocol.Range
}

type testFns struct {
	Tests      []testFn
	Benchmarks []testFn
}

func TestsAndBenchmarks(ctx context.Context, snapshot Snapshot, fh FileHandle) (testFns, error) {
	var out testFns

	if !strings.HasSuffix(fh.URI().Filename(), "_test.go") {
		return out, nil
	}
	pkg, pgf, err := GetParsedFile(ctx, snapshot, fh, WidestPackage)
	if err != nil {
		return out, err
	}

	for _, d := range pgf.File.Decls {
		fn, ok := d.(*ast.FuncDecl)
		if !ok {
			continue
		}

		rng, err := NewMappedRange(snapshot.FileSet(), pgf.Mapper, d.Pos(), fn.End()).Range()
		if err != nil {
			return out, err
		}

		if matchTestFunc(fn, pkg, testRe, "T") {
			out.Tests = append(out.Tests, testFn{fn.Name.Name, rng})
		}

		if matchTestFunc(fn, pkg, benchmarkRe, "B") {
			out.Benchmarks = append(out.Benchmarks, testFn{fn.Name.Name, rng})
		}
	}

	return out, nil
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
			rng, err := NewMappedRange(snapshot.FileSet(), pgf.Mapper, l.Pos(), l.Pos()+token.Pos(len(ggDirective))).Range()
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
						Command:   CommandGenerate.ID(),
						Arguments: nonRecursiveArgs,
					},
				},
				{
					Range: rng,
					Command: protocol.Command{
						Title:     "run go generate ./...",
						Command:   CommandGenerate.ID(),
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
	rng, err := NewMappedRange(snapshot.FileSet(), pgf.Mapper, c.Pos(), c.EndPos).Range()
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
				Command:   CommandRegenerateCgo.ID(),
				Arguments: jsonArgs,
			},
		},
	}, nil
}

func toggleDetailsCodeLens(ctx context.Context, snapshot Snapshot, fh FileHandle) ([]protocol.CodeLens, error) {
	_, pgf, err := GetParsedFile(ctx, snapshot, fh, WidestPackage)
	if err != nil {
		return nil, err
	}
	rng, err := NewMappedRange(snapshot.FileSet(), pgf.Mapper, pgf.File.Package, pgf.File.Package).Range()
	if err != nil {
		return nil, err
	}
	jsonArgs, err := MarshalArgs(fh.URI())
	if err != nil {
		return nil, err
	}
	return []protocol.CodeLens{{
		Range: rng,
		Command: protocol.Command{
			Title:     "Toggle gc annotation details",
			Command:   CommandToggleDetails.ID(),
			Arguments: jsonArgs,
		},
	}}, nil
}
