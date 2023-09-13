// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package inline_test

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/gob"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"strings"
	"testing"
	"unsafe"

	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/go/expect"
	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/go/types/typeutil"
	"golang.org/x/tools/internal/diff"
	"golang.org/x/tools/internal/refactor/inline"
	"golang.org/x/tools/internal/testenv"
	"golang.org/x/tools/txtar"
)

// TestData executes test scenarios specified by files in testdata/*.txtar.
func TestData(t *testing.T) {
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
						if err := doInlineNote(t.Logf, pkg, file, content, pattern, posn, want); err != nil {
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
func doInlineNote(logf func(string, ...any), pkg *packages.Package, file *ast.File, content []byte, pattern *regexp.Regexp, posn token.Position, want any) error {
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

		if err := checkTranscode(callee); err != nil {
			return nil, err
		}

		check := checkNoMutation(caller.File)
		defer check()
		return inline.Inline(logf, caller, callee)
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

// TestTable is a table driven test, enabling more compact expression
// of single-package test cases than is possible with the txtar notation.
func TestTable(t *testing.T) {
	// Each callee must declare a function or method named f,
	// and each caller must call it.
	const funcName = "f"

	var tests = []struct {
		descr          string
		callee, caller string // Go source files (sans package decl) of caller, callee
		want           string // expected new portion of caller file, or "error: regexp"
	}{
		{
			"Basic",
			`func f(x int) int { return x }`,
			`var _ = f(0)`,
			`var _ = (0)`,
		},
		{
			"Empty body, no arg effects.",
			`func f(x, y int) {}`,
			`func _() { f(1, 2) }`,
			`func _() {}`,
		},
		{
			"Empty body, some arg effects.",
			`func f(x, y, z int) {}`,
			`func _() { f(1, recover().(int), 3) }`,
			`func _() { _ = recover().(int) }`,
		},
		{
			"Tail call.",
			`func f() int { return 1 }`,
			`func _() int { return f() }`,
			`func _() int { return (1) }`,
		},
		{
			"Void tail call.",
			`func f() { println() }`,
			`func _() { f() }`,
			`func _() { println() }`,
		},
		{
			"Void tail call with defer.", // => literalized
			`func f() { defer f(); println() }`,
			`func _() { f() }`,
			`func _() { func() { defer f(); println() }() }`,
		},
		{
			"Edge case: cannot literalize spread method call.",
			`type I int
 			func g() (I, I)
			func (r I) f(x, y I) I {
				defer g() // force literalization
				return x + y + r
			}`,
			`func _() I { return recover().(I).f(g()) }`,
			`error: can't yet inline spread call to method`,
		},
		{
			"Variadic cancellation (basic).",
			`func f(args ...any) { defer f(&args); println(args) }`,
			`func _(slice []any) { f(slice...) }`,
			`func _(slice []any) { func(args []any) { defer f(&args); println(args) }(slice) }`,
		},
		{
			"Variadic cancellation (literalization with parameter elimination).",
			`func f(args ...any) { defer f(); println(args) }`,
			`func _(slice []any) { f(slice...) }`,
			`func _(slice []any) { func() { defer f(); println(slice) }() }`,
		},
		{
			"Variadic cancellation (reduction).",
			`func f(args ...any) { println(args) }`,
			`func _(slice []any) { f(slice...) }`,
			`func _(slice []any) { println(slice) }`,
		},
		{
			"Variadic elimination (literalization).",
			`func f(x any, rest ...any) { defer println(x, rest) }`, // defer => literalization
			`func _() { f(1, 2, 3) }`,
			`func _() { func(x any) { defer println(x, []any{2, 3}) }(1) }`,
		},
		{
			"Variadic elimination (reduction).",
			`func f(x int, rest ...int) { println(x, rest) }`,
			`func _() { f(1, 2, 3) }`,
			`func _() { println(1, []int{2, 3}) }`,
		},
		{
			"Spread call to variadic (1 arg, 1 param).",
			`func f(rest ...int) { println(rest) }; func g() (a, b int)`,
			`func _() { f(g()) }`,
			`func _() { func(rest ...int) { println(rest) }(g()) }`,
		},
		{
			"Spread call to variadic (1 arg, 2 params).",
			`func f(x int, rest ...int) { println(x, rest) }; func g() (a, b int)`,
			`func _() { f(g()) }`,
			`func _() { func(x int, rest ...int) { println(x, rest) }(g()) }`,
		},
		{
			"Spread call to variadic (1 arg, 3 params).",
			`func f(x, y int, rest ...int) { println(x, y, rest) }; func g() (a, b, c int)`,
			`func _() { f(g()) }`,
			`func _() { func(x, y int, rest ...int) { println(x, y, rest) }(g()) }`,
		},
		{
			"Binding declaration (x eliminated).",
			`func f(w, x, y any, z int) { println(w, y, z) }; func g(int) int`,
			`func _() { f(g(0), g(1), g(2), g(3)) }`,
			`func _() {
	{
		var (
			w, _, y any = g(0), g(1), g(2)
			z       int = g(3)
		)
		println(w, y, z)
	}
}`,
		},
		{
			"Binding decl in reduction of stmt-context call to { return exprs }",
			`func f(ch chan int) int { return <-ch }; func g() chan int`,
			`func _() { f(g()) }`,
			`func _() {
	{
		var ch chan int = g()
		<-ch
	}
}`,
		},
		{
			"No binding decl due to shadowing of int",
			`func f(int, y any, z int) { defer println(int, y, z) }; func g() int`,
			`func _() { f(g(), g(), g()) }`,
			`func _() { func(int, y any, z int) { defer println(int, y, z) }(g(), g(), g()) }
`,
		},
		// TODO(adonovan): improve coverage of the cross
		// product of each strategy with the checklist of
		// concerns enumerated in the package doc comment.
	}
	for _, test := range tests {
		test := test
		t.Run(test.descr, func(t *testing.T) {
			fset := token.NewFileSet()
			mustParse := func(filename string, content any) *ast.File {
				f, err := parser.ParseFile(fset, filename, content, parser.ParseComments|parser.SkipObjectResolution)
				if err != nil {
					t.Fatalf("ParseFile: %v", err)
				}
				return f
			}

			// Parse callee file and find first func decl named f.
			calleeContent := "package p\n" + test.callee
			calleeFile := mustParse("callee.go", calleeContent)
			var decl *ast.FuncDecl
			for _, d := range calleeFile.Decls {
				if d, ok := d.(*ast.FuncDecl); ok && d.Name.Name == funcName {
					decl = d
					break
				}
			}
			if decl == nil {
				t.Fatalf("declaration of func %s not found: %s", funcName, test.callee)
			}

			// Parse caller file and find first call to f().
			callerContent := "package p\n" + test.caller
			callerFile := mustParse("caller.go", callerContent)
			var call *ast.CallExpr
			ast.Inspect(callerFile, func(n ast.Node) bool {
				if n, ok := n.(*ast.CallExpr); ok {
					switch fun := n.Fun.(type) {
					case *ast.SelectorExpr:
						if fun.Sel.Name == funcName {
							call = n
						}
					case *ast.Ident:
						if fun.Name == funcName {
							call = n
						}
					}
				}
				return call == nil
			})
			if call == nil {
				t.Fatalf("call to %s not found: %s", funcName, test.caller)
			}

			// Type check both files as one package.
			info := &types.Info{
				Defs:       make(map[*ast.Ident]types.Object),
				Uses:       make(map[*ast.Ident]types.Object),
				Types:      make(map[ast.Expr]types.TypeAndValue),
				Implicits:  make(map[ast.Node]types.Object),
				Selections: make(map[*ast.SelectorExpr]*types.Selection),
				Scopes:     make(map[ast.Node]*types.Scope),
			}
			conf := &types.Config{Error: func(err error) { t.Error(err) }}
			pkg, err := conf.Check("p", fset, []*ast.File{callerFile, calleeFile}, info)
			if err != nil {
				t.Fatal(err)
			}

			// Analyze callee and inline call.
			doIt := func() ([]byte, error) {
				callee, err := inline.AnalyzeCallee(fset, pkg, info, decl, []byte(calleeContent))
				if err != nil {
					return nil, err
				}
				if err := checkTranscode(callee); err != nil {
					t.Fatal(err)
				}

				caller := &inline.Caller{
					Fset:    fset,
					Types:   pkg,
					Info:    info,
					File:    callerFile,
					Call:    call,
					Content: []byte(callerContent),
				}
				check := checkNoMutation(caller.File)
				defer check()
				return inline.Inline(t.Logf, caller, callee)
			}
			gotContent, err := doIt()

			// Want error?
			if rest := strings.TrimPrefix(test.want, "error: "); rest != test.want {
				if err == nil {
					t.Fatalf("unexpected sucess: want error matching %q", rest)
				}
				msg := err.Error()
				if ok, err := regexp.MatchString(rest, msg); err != nil {
					t.Fatalf("invalid regexp: %v", err)
				} else if !ok {
					t.Fatalf("wrong error: %s (want match for %q)", msg, rest)
				}
				return
			}

			// Want success.
			if err != nil {
				t.Fatal(err)
			}

			// Compute a single-hunk line-based diff.
			srcLines := strings.Split(callerContent, "\n")
			gotLines := strings.Split(string(gotContent), "\n")
			for len(srcLines) > 0 && len(gotLines) > 0 &&
				srcLines[0] == gotLines[0] {
				srcLines = srcLines[1:]
				gotLines = gotLines[1:]
			}
			for len(srcLines) > 0 && len(gotLines) > 0 &&
				srcLines[len(srcLines)-1] == gotLines[len(gotLines)-1] {
				srcLines = srcLines[:len(srcLines)-1]
				gotLines = gotLines[:len(gotLines)-1]
			}
			got := strings.Join(gotLines, "\n")

			if strings.TrimSpace(got) != strings.TrimSpace(test.want) {
				t.Fatalf("\nInlining this call:\t%s\nof this callee:    \t%s\nproduced:\n%s\nWant:\n\n%s",
					test.caller,
					test.callee,
					got,
					test.want)
			}

			// Check that resulting code type-checks.
			newCallerFile := mustParse("newcaller.go", gotContent)
			if _, err := conf.Check("p", fset, []*ast.File{newCallerFile, calleeFile}, nil); err != nil {
				t.Fatalf("modified source failed to typecheck: <<%s>>", gotContent)
			}
		})
	}
}

// -- helpers --

// checkNoMutation returns a function that, when called,
// asserts that file was not modified since the checkNoMutation call.
func checkNoMutation(file *ast.File) func() {
	pre := deepHash(file)
	return func() {
		post := deepHash(file)
		if pre != post {
			panic("Inline mutated caller.File")
		}
	}
}

// checkTranscode replaces *callee by the results of gob-encoding and
// then decoding it, to test that these operations are lossless.
func checkTranscode(callee *inline.Callee) error {
	// Perform Gob transcoding so that it is exercised by the test.
	var enc bytes.Buffer
	if err := gob.NewEncoder(&enc).Encode(callee); err != nil {
		return fmt.Errorf("internal error: gob encoding failed: %v", err)
	}
	*callee = inline.Callee{}
	if err := gob.NewDecoder(&enc).Decode(callee); err != nil {
		return fmt.Errorf("internal error: gob decoding failed: %v", err)
	}
	return nil
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

// deepHash computes a cryptographic hash of an ast.Node so that
// if the data structure is mutated, the hash changes.
// It assumes Go variables do not change address.
//
// TODO(adonovan): consider publishing this in the astutil package.
//
// TODO(adonovan): consider a variant that reports where in the tree
// the mutation occurred (obviously at a cost in space).
func deepHash(n ast.Node) any {
	seen := make(map[unsafe.Pointer]bool) // to break cycles

	hasher := sha256.New()
	le := binary.LittleEndian
	writeUint64 := func(v uint64) {
		var bs [8]byte
		le.PutUint64(bs[:], v)
		hasher.Write(bs[:])
	}

	var visit func(reflect.Value)
	visit = func(v reflect.Value) {
		switch v.Kind() {
		case reflect.Ptr:
			ptr := v.UnsafePointer()
			writeUint64(uint64(uintptr(ptr)))
			if !v.IsNil() {
				if !seen[ptr] {
					seen[ptr] = true
					// Skip types we don't handle yet, but don't care about.
					switch v.Interface().(type) {
					case *ast.Scope:
						return // involves a map
					}

					visit(v.Elem())
				}
			}

		case reflect.Struct:
			for i := 0; i < v.Type().NumField(); i++ {
				visit(v.Field(i))
			}

		case reflect.Slice:
			ptr := v.UnsafePointer()
			// We may encounter different slices at the same address,
			// so don't mark ptr as "seen".
			writeUint64(uint64(uintptr(ptr)))
			writeUint64(uint64(v.Len()))
			writeUint64(uint64(v.Cap()))
			for i := 0; i < v.Len(); i++ {
				visit(v.Index(i))
			}

		case reflect.Interface:
			if v.IsNil() {
				writeUint64(0)
			} else {
				rtype := reflect.ValueOf(v.Type()).UnsafePointer()
				writeUint64(uint64(uintptr(rtype)))
				visit(v.Elem())
			}

		case reflect.Array, reflect.Chan, reflect.Func, reflect.Map, reflect.UnsafePointer:
			panic(v) // unreachable in AST

		default: // bool, string, number
			if v.Kind() == reflect.String { // proper framing
				writeUint64(uint64(v.Len()))
			}
			binary.Write(hasher, le, v.Interface())
		}
	}
	visit(reflect.ValueOf(n))

	var hash [sha256.Size]byte
	hasher.Sum(hash[:0])
	return hash
}
