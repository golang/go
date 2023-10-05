// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package inline_test

import (
	"flag"
	"fmt"
	"go/ast"
	"go/parser"
	"go/types"
	"log"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/go/types/typeutil"
	"golang.org/x/tools/internal/diff"
	"golang.org/x/tools/internal/refactor/inline"
	"golang.org/x/tools/internal/testenv"
)

var packagesFlag = flag.String("packages", "", "set of packages for TestEverything")

// TestEverything invokes the inliner on every single call site in a
// given package. and checks that it produces either a reasonable
// error, or output that parses and type-checks.
//
// It does nothing during ordinary testing, but may be used to find
// inlining bugs in large corpora.
//
// Use this command to inline everything in golang.org/x/tools:
//
// $ go test ./internal/refactor/inline/ -run=Everything -packages=../../../
//
// And these commands to inline everything in the kubernetes repository:
//
// $ go test -c -o /tmp/everything ./internal/refactor/inline/
// $ (cd kubernetes && /tmp/everything -test.run=Everything -packages=./...)
//
// TODO(adonovan):
//   - report counters (number of attempts, failed AnalyzeCallee, failed
//     Inline, etc.)
//   - Make a pretty log of the entire output so that we can peruse it
//     for opportunities for systematic improvement.
func TestEverything(t *testing.T) {
	testenv.NeedsGoPackages(t)
	if testing.Short() {
		t.Skipf("skipping slow test in -short mode")
	}
	if *packagesFlag == "" {
		return
	}

	// Load this package plus dependencies from typed syntax.
	cfg := &packages.Config{
		Mode: packages.LoadAllSyntax,
		Env: append(os.Environ(),
			"GO111MODULES=on",
			"GOPATH=",
			"GOWORK=off",
			"GOPROXY=off"),
	}
	pkgs, err := packages.Load(cfg, *packagesFlag)
	if err != nil {
		t.Errorf("Load: %v", err)
	}
	// Report parse/type errors.
	// Also, build transitive dependency mapping.
	deps := make(map[string]*packages.Package) // key is PkgPath
	packages.Visit(pkgs, nil, func(pkg *packages.Package) {
		deps[pkg.Types.Path()] = pkg
		for _, err := range pkg.Errors {
			t.Fatal(err)
		}
	})

	// Memoize repeated calls for same file.
	fileContent := make(map[string][]byte)
	readFile := func(filename string) ([]byte, error) {
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

	for _, callerPkg := range pkgs {
		// Find all static function calls in the package.
		for _, callerFile := range callerPkg.Syntax {
			noMutCheck := checkNoMutation(callerFile)
			ast.Inspect(callerFile, func(n ast.Node) bool {
				call, ok := n.(*ast.CallExpr)
				if !ok {
					return true
				}
				fn := typeutil.StaticCallee(callerPkg.TypesInfo, call)
				if fn == nil {
					return true
				}

				// Prepare caller info.
				callPosn := callerPkg.Fset.PositionFor(call.Lparen, false)
				callerContent, err := readFile(callPosn.Filename)
				if err != nil {
					t.Fatal(err)
				}
				caller := &inline.Caller{
					Fset:    callerPkg.Fset,
					Types:   callerPkg.Types,
					Info:    callerPkg.TypesInfo,
					File:    callerFile,
					Call:    call,
					Content: callerContent,
				}

				// Analyze callee.
				calleePkg, ok := deps[fn.Pkg().Path()]
				if !ok {
					t.Fatalf("missing package for callee %v", fn)
				}
				calleePosn := callerPkg.Fset.PositionFor(fn.Pos(), false)
				calleeDecl, err := findFuncByPosition(calleePkg, calleePosn)
				if err != nil {
					t.Fatal(err)
				}
				calleeContent, err := readFile(calleePosn.Filename)
				if err != nil {
					t.Fatal(err)
				}

				// Create a subtest for each inlining operation.
				name := fmt.Sprintf("%s@%v", fn.Name(), filepath.Base(callPosn.String()))
				t.Run(name, func(t *testing.T) {
					// TODO(adonovan): add a panic handler.

					t.Logf("callee declared at %v",
						filepath.Base(calleePosn.String()))

					t.Logf("run this command to reproduce locally:\n$ gopls fix -a -d %s:#%d refactor.inline",
						callPosn.Filename, callPosn.Offset)

					callee, err := inline.AnalyzeCallee(
						t.Logf,
						calleePkg.Fset,
						calleePkg.Types,
						calleePkg.TypesInfo,
						calleeDecl,
						calleeContent)
					if err != nil {
						// Ignore the expected kinds of errors.
						for _, ignore := range []string{
							"has no body",
							"type parameters are not yet",
							"line directives",
							"cgo-generated",
						} {
							if strings.Contains(err.Error(), ignore) {
								return
							}
						}
						t.Fatalf("AnalyzeCallee: %v", err)
					}
					if err := checkTranscode(callee); err != nil {
						t.Fatal(err)
					}

					got, err := inline.Inline(t.Logf, caller, callee)
					if err != nil {
						// Write error to a log, but this ok.
						t.Log(err)
						return
					}

					// Print the diff.
					t.Logf("Got diff:\n%s",
						diff.Unified("old", "new", string(callerContent), string(got)))

					// Parse and type-check the transformed source.
					f, err := parser.ParseFile(caller.Fset, callPosn.Filename, got, parser.SkipObjectResolution)
					if err != nil {
						t.Fatalf("transformed source does not parse: %v", err)
					}
					// Splice into original file list.
					syntax := append([]*ast.File(nil), callerPkg.Syntax...)
					for i := range callerPkg.Syntax {
						if syntax[i] == callerFile {
							syntax[i] = f
							break
						}
					}

					var typeErrors []string
					conf := &types.Config{
						Error: func(err error) {
							typeErrors = append(typeErrors, err.Error())
						},
						Importer: importerFunc(func(importPath string) (*types.Package, error) {
							// Note: deps is properly keyed by package path,
							// not import path, but we can't assume
							// Package.Imports[importPath] exists in the
							// case of newly added imports of indirect
							// dependencies. Seems not to matter to this test.
							dep, ok := deps[importPath]
							if ok {
								return dep.Types, nil
							}
							return nil, fmt.Errorf("missing package: %q", importPath)
						}),
					}
					if _, err := conf.Check("p", caller.Fset, syntax, nil); err != nil {
						t.Fatalf("transformed package has type errors:\n\n%s\n\nTransformed file:\n\n%s",
							strings.Join(typeErrors, "\n"),
							got)
					}
				})
				return true
			})
			noMutCheck()
		}
	}
	log.Printf("Analyzed %d packages", len(pkgs))
}

type importerFunc func(path string) (*types.Package, error)

func (f importerFunc) Import(path string) (*types.Package, error) {
	return f(path)
}
