// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package inline_test

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"go/ast"
	"go/token"
	"os"
	"path/filepath"
	"regexp"
	"testing"

	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/go/expect"
	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/go/types/typeutil"
	"golang.org/x/tools/internal/diff"
	"golang.org/x/tools/internal/refactor/inline"
	"golang.org/x/tools/internal/testenv"
	"golang.org/x/tools/txtar"
)

// Test executes test scenarios specified by files in testdata/*.txtar.
func Test(t *testing.T) {
	testenv.NeedsGoPackages(t)

	files, err := filepath.Glob("testdata/*.txtar")
	if err != nil {
		t.Fatal(err)
	}
	for _, file := range files {
		file := file
		t.Run(filepath.Base(file), func(t *testing.T) {
			t.Parallel()

			// Extract archive to temporary tree.
			ar, err := txtar.ParseFile(file)
			if err != nil {
				t.Fatal(err)
			}
			dir := t.TempDir()
			if err := extractTxtar(ar, dir); err != nil {
				t.Fatal(err)
			}

			// Load packages.
			cfg := &packages.Config{
				Dir:  dir,
				Mode: packages.LoadAllSyntax,
				Env: append(os.Environ(),
					"GO111MODULES=on",
					"GOPATH=",
					"GOWORK=off",
					"GOPROXY=off"),
			}
			pkgs, err := packages.Load(cfg, "./...")
			if err != nil {
				t.Errorf("Load: %v", err)
			}
			// Report parse/type errors; they may be benign.
			packages.Visit(pkgs, nil, func(pkg *packages.Package) {
				for _, err := range pkg.Errors {
					t.Log(err)
				}
			})

			// Process @inline notes in comments in initial packages.
			for _, pkg := range pkgs {
				for _, file := range pkg.Syntax {
					// Read file content (for @inline regexp, and inliner).
					content, err := os.ReadFile(pkg.Fset.File(file.Pos()).Name())
					if err != nil {
						t.Error(err)
						continue
					}

					// Read and process @inline notes.
					notes, err := expect.ExtractGo(pkg.Fset, file)
					if err != nil {
						t.Errorf("parsing notes in %q: %v", pkg.Fset.File(file.Pos()).Name(), err)
						continue
					}
					for _, note := range notes {
						posn := pkg.Fset.Position(note.Pos)
						if note.Name != "inline" {
							t.Errorf("%s: invalid marker @%s", posn, note.Name)
							continue
						}
						if nargs := len(note.Args); nargs != 2 {
							t.Errorf("@inline: want 2 args, got %d", nargs)
							continue
						}
						pattern, ok := note.Args[0].(*regexp.Regexp)
						if !ok {
							t.Errorf("%s: @inline(rx, want): want regular expression rx", posn)
							continue
						}

						// want is a []byte (success) or *Regexp (failure)
						var want any
						switch x := note.Args[1].(type) {
						case string, expect.Identifier:
							for _, file := range ar.Files {
								if file.Name == fmt.Sprint(x) {
									want = file.Data
									break
								}
							}
							if want == nil {
								t.Errorf("%s: @inline(rx, want): archive entry %q not found", posn, x)
								continue
							}
						case *regexp.Regexp:
							want = x
						default:
							t.Errorf("%s: @inline(rx, want): want file name (to assert success) or error message regexp (to assert failure)", posn)
							continue
						}
						t.Log("doInlineNote", posn)
						if err := doInlineNote(pkg, file, content, pattern, posn, want); err != nil {
							t.Errorf("%s: @inline(%v, %v): %v", posn, note.Args[0], note.Args[1], err)
							continue
						}
					}
				}
			}
		})
	}
}

// doInlineNote executes an assertion specified by a single
// @inline(re"pattern", want) note in a comment. It finds the first
// match of regular expression 'pattern' on the same line, finds the
// innermost enclosing CallExpr, and inlines it.
//
// Finally it checks that, on success, the transformed file is equal
// to want (a []byte), or on failure that the error message matches
// want (a *Regexp).
func doInlineNote(pkg *packages.Package, file *ast.File, content []byte, pattern *regexp.Regexp, posn token.Position, want any) error {
	// Find extent of pattern match within commented line.
	var startPos, endPos token.Pos
	{
		tokFile := pkg.Fset.File(file.Pos())
		lineStartOffset := int(tokFile.LineStart(posn.Line)) - tokFile.Base()
		line := content[lineStartOffset:]
		if i := bytes.IndexByte(line, '\n'); i >= 0 {
			line = line[:i]
		}
		matches := pattern.FindSubmatchIndex(line)
		var start, end int // offsets
		switch len(matches) {
		case 2:
			// no subgroups: return the range of the regexp expression
			start, end = matches[0], matches[1]
		case 4:
			// one subgroup: return its range
			start, end = matches[2], matches[3]
		default:
			return fmt.Errorf("invalid location regexp %q: expect either 0 or 1 subgroups, got %d",
				pattern, len(matches)/2-1)
		}
		startPos = tokFile.Pos(lineStartOffset + start)
		endPos = tokFile.Pos(lineStartOffset + end)
	}

	// Find innermost call enclosing the pattern match.
	var caller *inline.Caller
	{
		path, _ := astutil.PathEnclosingInterval(file, startPos, endPos)
		for _, n := range path {
			if call, ok := n.(*ast.CallExpr); ok {
				caller = &inline.Caller{
					Fset:    pkg.Fset,
					Types:   pkg.Types,
					Info:    pkg.TypesInfo,
					File:    file,
					Call:    call,
					Content: content,
				}
				break
			}
		}
		if caller == nil {
			return fmt.Errorf("no enclosing call")
		}
	}

	// Is it a static function call?
	fn := typeutil.StaticCallee(caller.Info, caller.Call)
	if fn == nil {
		return fmt.Errorf("cannot inline: not a static call")
	}

	// Find callee function.
	var (
		calleePkg  *packages.Package
		calleeDecl *ast.FuncDecl
	)
	{
		var same func(*ast.FuncDecl) bool
		// Is the call within the package?
		if fn.Pkg() == caller.Types {
			calleePkg = pkg // same as caller
			same = func(decl *ast.FuncDecl) bool {
				return decl.Name.Pos() == fn.Pos()
			}
		} else {
			// Different package. Load it now.
			// (The primary load loaded all dependencies,
			// but we choose to load it again, with
			// a distinct token.FileSet and types.Importer,
			// to keep the implementation honest.)
			cfg := &packages.Config{
				// TODO(adonovan): get the original module root more cleanly
				Dir:  filepath.Dir(filepath.Dir(pkg.GoFiles[0])),
				Fset: token.NewFileSet(),
				Mode: packages.LoadSyntax,
			}
			roots, err := packages.Load(cfg, fn.Pkg().Path())
			if err != nil {
				return fmt.Errorf("loading callee package: %v", err)
			}
			if packages.PrintErrors(roots) > 0 {
				return fmt.Errorf("callee package had errors") // (see log)
			}
			calleePkg = roots[0]
			posn := caller.Fset.Position(fn.Pos()) // callee posn wrt caller package
			same = func(decl *ast.FuncDecl) bool {
				// We can't rely on columns in export data:
				// some variants replace it with 1.
				// We can't expect file names to have the same prefix.
				// export data for go1.20 std packages have  $GOROOT written in
				// them, so how are we supposed to find the source? Yuck!
				// Ugh. need to samefile? Nope $GOROOT just won't work
				// This is highly client specific anyway.
				posn2 := calleePkg.Fset.Position(decl.Name.Pos())
				return posn.Filename == posn2.Filename &&
					posn.Line == posn2.Line
			}
		}

		for _, file := range calleePkg.Syntax {
			for _, decl := range file.Decls {
				if decl, ok := decl.(*ast.FuncDecl); ok && same(decl) {
					calleeDecl = decl
					goto found
				}
			}
		}
		return fmt.Errorf("can't find FuncDecl for callee") // can't happen?
	found:
	}

	// Do the inlining. For the purposes of the test,
	// AnalyzeCallee and Inline are a single operation.
	got, err := func() ([]byte, error) {
		filename := calleePkg.Fset.File(calleeDecl.Pos()).Name()
		content, err := os.ReadFile(filename)
		if err != nil {
			return nil, err
		}
		callee, err := inline.AnalyzeCallee(
			calleePkg.Fset,
			calleePkg.Types,
			calleePkg.TypesInfo,
			calleeDecl,
			content)
		if err != nil {
			return nil, err
		}

		// Perform Gob transcoding so that it is exercised by the test.
		var enc bytes.Buffer
		if err := gob.NewEncoder(&enc).Encode(callee); err != nil {
			return nil, fmt.Errorf("internal error: gob encoding failed: %v", err)
		}
		*callee = inline.Callee{}
		if err := gob.NewDecoder(&enc).Decode(callee); err != nil {
			return nil, fmt.Errorf("internal error: gob decoding failed: %v", err)
		}

		return inline.Inline(caller, callee)
	}()
	if err != nil {
		if wantRE, ok := want.(*regexp.Regexp); ok {
			if !wantRE.MatchString(err.Error()) {
				return fmt.Errorf("Inline failed with wrong error: %v (want error matching %q)", err, want)
			}
			return nil // expected error
		}
		return fmt.Errorf("Inline failed: %v", err) // success was expected
	}

	// Inline succeeded.
	if want, ok := want.([]byte); ok {
		got = append(bytes.TrimSpace(got), '\n')
		want = append(bytes.TrimSpace(want), '\n')
		if diff := diff.Unified("want", "got", string(want), string(got)); diff != "" {
			return fmt.Errorf("Inline returned wrong output:\n%s\nWant:\n%s\nDiff:\n%s",
				got, want, diff)
		}
		return nil
	}
	return fmt.Errorf("Inline succeeded unexpectedly: want error matching %q, got <<%s>>", want, got)

}

// TODO(adonovan): publish this a helper (#61386).
func extractTxtar(ar *txtar.Archive, dir string) error {
	for _, file := range ar.Files {
		name := filepath.Join(dir, file.Name)
		if err := os.MkdirAll(filepath.Dir(name), 0777); err != nil {
			return err
		}
		if err := os.WriteFile(name, file.Data, 0666); err != nil {
			return err
		}
	}
	return nil
}
