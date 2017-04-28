// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements TestFormats; a test that verifies
// format strings in the compiler (this directory and all
// subdirectories, recursively).
//
// TestFormats finds potential (Printf, etc.) format strings.
// If they are used in a call, the format verbs are verified
// based on the matching argument type against a precomputed
// table of valid formats. The knownFormats table can be used
// to automatically rewrite format strings with the -u flag.
//
// A new knownFormats table based on the found formats is printed
// when the test is run in verbose mode (-v flag). The table
// needs to be updated whenever a new (type, format) combination
// is found and the format verb is not 'v' or 'T' (as in "%v" or
// "%T").
//
// Run as: go test -run Formats [-u][-v]
//
// Known bugs:
// - indexed format strings ("%[2]s", etc.) are not supported
//   (the test will fail)
// - format strings that are not simple string literals cannot
//   be updated automatically
//   (the test will fail with respective warnings)
// - format strings in _test packages outside the current
//   package are not processed
//   (the test will report those files)
//
package main_test

import (
	"bytes"
	"flag"
	"fmt"
	"go/ast"
	"go/build"
	"go/constant"
	"go/format"
	"go/importer"
	"go/parser"
	"go/token"
	"go/types"
	"internal/testenv"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"
	"unicode/utf8"
)

var update = flag.Bool("u", false, "update format strings")

// The following variables collect information across all processed files.
var (
	fset          = token.NewFileSet()
	formatStrings = make(map[*ast.BasicLit]bool)      // set of all potential format strings found
	foundFormats  = make(map[string]bool)             // set of all formats found
	callSites     = make(map[*ast.CallExpr]*callSite) // map of all calls
)

// A File is a corresponding (filename, ast) pair.
type File struct {
	name string
	ast  *ast.File
}

func TestFormats(t *testing.T) {
	testenv.MustHaveGoBuild(t) // more restrictive than necessary, but that's ok

	// process all directories
	filepath.Walk(".", func(path string, info os.FileInfo, err error) error {
		if info.IsDir() {
			if info.Name() == "testdata" {
				return filepath.SkipDir
			}

			importPath := filepath.Join("cmd/compile", path)
			if blacklistedPackages[filepath.ToSlash(importPath)] {
				return filepath.SkipDir
			}

			pkg, err := build.Import(importPath, path, 0)
			if err != nil {
				if _, ok := err.(*build.NoGoError); ok {
					return nil // nothing to do here
				}
				t.Fatal(err)
			}
			collectPkgFormats(t, pkg)
		}
		return nil
	})

	// test and rewrite formats
	updatedFiles := make(map[string]File) // files that were rewritten
	for _, p := range callSites {
		// test current format literal and determine updated one
		out := formatReplace(p.str, func(index int, in string) string {
			if in == "*" {
				return in // cannot rewrite '*' (as in "%*d")
			}
			// in != '*'
			typ := p.types[index]
			format := typ + " " + in // e.g., "*Node %n"

			// check if format is known
			out, known := knownFormats[format]

			// record format if not yet found
			_, found := foundFormats[format]
			if !found {
				foundFormats[format] = true
			}

			// report an error if the format is unknown and this is the first
			// time we see it; ignore "%v" and "%T" which are always valid
			if !known && !found && in != "%v" && in != "%T" {
				t.Errorf("%s: unknown format %q for %s argument", posString(p.arg), in, typ)
			}

			if out == "" {
				out = in
			}
			return out
		})

		// replace existing format literal if it changed
		if out != p.str {
			// we cannot replace the argument if it's not a string literal for now
			// (e.g., it may be "foo" + "bar")
			lit, ok := p.arg.(*ast.BasicLit)
			if !ok {
				delete(callSites, p.call) // treat as if we hadn't found this site
				continue
			}

			if testing.Verbose() {
				fmt.Printf("%s:\n\t- %q\n\t+ %q\n", posString(p.arg), p.str, out)
			}

			// find argument index of format argument
			index := -1
			for i, arg := range p.call.Args {
				if p.arg == arg {
					index = i
					break
				}
			}
			if index < 0 {
				// we may have processed the same call site twice,
				// but that shouldn't happen
				panic("internal error: matching argument not found")
			}

			// replace literal
			new := *lit                    // make a copy
			new.Value = strconv.Quote(out) // this may introduce "-quotes where there were `-quotes
			p.call.Args[index] = &new
			updatedFiles[p.file.name] = p.file
		}
	}

	// write dirty files back
	var filesUpdated bool
	if len(updatedFiles) > 0 && *update {
		for _, file := range updatedFiles {
			var buf bytes.Buffer
			if err := format.Node(&buf, fset, file.ast); err != nil {
				t.Errorf("WARNING: formatting %s failed: %v", file.name, err)
				continue
			}
			if err := ioutil.WriteFile(file.name, buf.Bytes(), 0x666); err != nil {
				t.Errorf("WARNING: writing %s failed: %v", file.name, err)
				continue
			}
			fmt.Printf("updated %s\n", file.name)
			filesUpdated = true
		}
	}

	// report all function names containing a format string
	if len(callSites) > 0 && testing.Verbose() {
		set := make(map[string]bool)
		for _, p := range callSites {
			set[nodeString(p.call.Fun)] = true
		}
		var list []string
		for s := range set {
			list = append(list, s)
		}
		fmt.Println("\nFunctions")
		printList(list)
	}

	// report all formats found
	if len(foundFormats) > 0 && testing.Verbose() {
		var list []string
		for s := range foundFormats {
			list = append(list, fmt.Sprintf("%q: \"\",", s))
		}
		fmt.Println("\nvar knownFormats = map[string]string{")
		printList(list)
		fmt.Println("}")
	}

	// check that knownFormats is up to date
	if !testing.Verbose() && !*update {
		var mismatch bool
		for s := range foundFormats {
			if _, ok := knownFormats[s]; !ok {
				mismatch = true
				break
			}
		}
		if !mismatch {
			for s := range knownFormats {
				if _, ok := foundFormats[s]; !ok {
					mismatch = true
					break
				}
			}
		}
		if mismatch {
			t.Errorf("knownFormats is out of date; please run with -v to regenerate")
		}
	}

	// all format strings of calls must be in the formatStrings set (self-verification)
	for _, p := range callSites {
		if lit, ok := p.arg.(*ast.BasicLit); ok && lit.Kind == token.STRING {
			if formatStrings[lit] {
				// ok
				delete(formatStrings, lit)
			} else {
				// this should never happen
				panic(fmt.Sprintf("internal error: format string not found (%s)", posString(lit)))
			}
		}
	}

	// if we have any strings left, we may need to update them manually
	if len(formatStrings) > 0 && filesUpdated {
		var list []string
		for lit := range formatStrings {
			list = append(list, fmt.Sprintf("%s: %s", posString(lit), nodeString(lit)))
		}
		fmt.Println("\nWARNING: Potentially missed format strings")
		printList(list)
		t.Fail()
	}

	fmt.Println()
}

// A callSite describes a function call that appears to contain
// a format string.
type callSite struct {
	file  File
	call  *ast.CallExpr // call containing the format string
	arg   ast.Expr      // format argument (string literal or constant)
	str   string        // unquoted format string
	types []string      // argument types
}

func collectPkgFormats(t *testing.T, pkg *build.Package) {
	// collect all files
	var filenames []string
	filenames = append(filenames, pkg.GoFiles...)
	filenames = append(filenames, pkg.CgoFiles...)
	filenames = append(filenames, pkg.TestGoFiles...)

	// TODO(gri) verify _test files outside package
	for _, name := range pkg.XTestGoFiles {
		// don't process this test itself
		if name != "fmt_test.go" && testing.Verbose() {
			fmt.Printf("WARNING: %s not processed\n", filepath.Join(pkg.Dir, name))
		}
	}

	// make filenames relative to .
	for i, name := range filenames {
		filenames[i] = filepath.Join(pkg.Dir, name)
	}

	// parse all files
	files := make([]*ast.File, len(filenames))
	for i, filename := range filenames {
		f, err := parser.ParseFile(fset, filename, nil, parser.ParseComments)
		if err != nil {
			t.Fatal(err)
		}
		files[i] = f
	}

	// typecheck package
	conf := types.Config{Importer: importer.Default()}
	etypes := make(map[ast.Expr]types.TypeAndValue)
	if _, err := conf.Check(pkg.ImportPath, fset, files, &types.Info{Types: etypes}); err != nil {
		t.Fatal(err)
	}

	// collect all potential format strings (for extra verification later)
	for _, file := range files {
		ast.Inspect(file, func(n ast.Node) bool {
			if s, ok := stringLit(n); ok && isFormat(s) {
				formatStrings[n.(*ast.BasicLit)] = true
			}
			return true
		})
	}

	// collect all formats/arguments of calls with format strings
	for index, file := range files {
		ast.Inspect(file, func(n ast.Node) bool {
			if call, ok := n.(*ast.CallExpr); ok {
				// ignore blacklisted functions
				if blacklistedFunctions[nodeString(call.Fun)] {
					return true
				}
				// look for an arguments that might be a format string
				for i, arg := range call.Args {
					if s, ok := stringVal(etypes[arg]); ok && isFormat(s) {
						// make sure we have enough arguments
						n := numFormatArgs(s)
						if i+1+n > len(call.Args) {
							t.Errorf("%s: not enough format args (blacklist %s?)", posString(call), nodeString(call.Fun))
							break // ignore this call
						}
						// assume last n arguments are to be formatted;
						// determine their types
						argTypes := make([]string, n)
						for i, arg := range call.Args[len(call.Args)-n:] {
							if tv, ok := etypes[arg]; ok {
								argTypes[i] = typeString(tv.Type)
							}
						}
						// collect call site
						if callSites[call] != nil {
							panic("internal error: file processed twice?")
						}
						callSites[call] = &callSite{
							file:  File{filenames[index], file},
							call:  call,
							arg:   arg,
							str:   s,
							types: argTypes,
						}
						break // at most one format per argument list
					}
				}
			}
			return true
		})
	}
}

// printList prints list in sorted order.
func printList(list []string) {
	sort.Strings(list)
	for _, s := range list {
		fmt.Println("\t", s)
	}
}

// posString returns a string representation of n's position
// in the form filename:line:col: .
func posString(n ast.Node) string {
	if n == nil {
		return ""
	}
	return fset.Position(n.Pos()).String()
}

// nodeString returns a string representation of n.
func nodeString(n ast.Node) string {
	var buf bytes.Buffer
	if err := format.Node(&buf, fset, n); err != nil {
		log.Fatal(err) // should always succeed
	}
	return buf.String()
}

// typeString returns a string representation of n.
func typeString(typ types.Type) string {
	return filepath.ToSlash(typ.String())
}

// stringLit returns the unquoted string value and true if
// n represents a string literal; otherwise it returns ""
// and false.
func stringLit(n ast.Node) (string, bool) {
	if lit, ok := n.(*ast.BasicLit); ok && lit.Kind == token.STRING {
		s, err := strconv.Unquote(lit.Value)
		if err != nil {
			log.Fatal(err) // should not happen with correct ASTs
		}
		return s, true
	}
	return "", false
}

// stringVal returns the (unquoted) string value and true if
// tv is a string constant; otherwise it returns "" and false.
func stringVal(tv types.TypeAndValue) (string, bool) {
	if tv.IsValue() && tv.Value != nil && tv.Value.Kind() == constant.String {
		return constant.StringVal(tv.Value), true
	}
	return "", false
}

// formatIter iterates through the string s in increasing
// index order and calls f for each format specifier '%..v'.
// The arguments for f describe the specifier's index range.
// If a format specifier contains a  "*", f is called with
// the index range for "*" alone, before being called for
// the entire specifier. The result of f is the index of
// the rune at which iteration continues.
func formatIter(s string, f func(i, j int) int) {
	i := 0     // index after current rune
	var r rune // current rune

	next := func() {
		r1, w := utf8.DecodeRuneInString(s[i:])
		if w == 0 {
			r1 = -1 // signal end-of-string
		}
		r = r1
		i += w
	}

	flags := func() {
		for r == ' ' || r == '#' || r == '+' || r == '-' || r == '0' {
			next()
		}
	}

	index := func() {
		if r == '[' {
			log.Fatalf("cannot handle indexed arguments: %s", s)
		}
	}

	digits := func() {
		index()
		if r == '*' {
			i = f(i-1, i)
			next()
			return
		}
		for '0' <= r && r <= '9' {
			next()
		}
	}

	for next(); r >= 0; next() {
		if r == '%' {
			i0 := i
			next()
			flags()
			digits()
			if r == '.' {
				next()
				digits()
			}
			index()
			// accept any letter (a-z, A-Z) as format verb;
			// ignore anything else
			if 'a' <= r && r <= 'z' || 'A' <= r && r <= 'Z' {
				i = f(i0-1, i)
			}
		}
	}
}

// isFormat reports whether s contains format specifiers.
func isFormat(s string) (yes bool) {
	formatIter(s, func(i, j int) int {
		yes = true
		return len(s) // stop iteration
	})
	return
}

// oneFormat reports whether s is exactly one format specifier.
func oneFormat(s string) (yes bool) {
	formatIter(s, func(i, j int) int {
		yes = i == 0 && j == len(s)
		return j
	})
	return
}

// numFormatArgs returns the number of format specifiers in s.
func numFormatArgs(s string) int {
	count := 0
	formatIter(s, func(i, j int) int {
		count++
		return j
	})
	return count
}

// formatReplace replaces the i'th format specifier s in the incoming
// string in with the result of f(i, s) and returns the new string.
func formatReplace(in string, f func(i int, s string) string) string {
	var buf []byte
	i0 := 0
	index := 0
	formatIter(in, func(i, j int) int {
		if sub := in[i:j]; sub != "*" { // ignore calls for "*" width/length specifiers
			buf = append(buf, in[i0:i]...)
			buf = append(buf, f(index, sub)...)
			i0 = j
		}
		index++
		return j
	})
	return string(append(buf, in[i0:]...))
}

// blacklistedPackages is the set of packages which can
// be ignored.
var blacklistedPackages = map[string]bool{}

// blacklistedFunctions is the set of functions which may have
// format-like arguments but which don't do any formatting and
// thus may be ignored.
var blacklistedFunctions = map[string]bool{}

func init() {
	// verify that knownFormats entries are correctly formatted
	for key, val := range knownFormats {
		// key must be "typename format", and format starts with a '%'
		// (formats containing '*' alone are not collected in this table)
		i := strings.Index(key, "%")
		if i < 0 || !oneFormat(key[i:]) {
			log.Fatalf("incorrect knownFormats key: %q", key)
		}
		// val must be "format" or ""
		if val != "" && !oneFormat(val) {
			log.Fatalf("incorrect knownFormats value: %q (key = %q)", val, key)
		}
	}
}

// knownFormats entries are of the form "typename format" -> "newformat".
// An absent entry means that the format is not recognized as valid.
// An empty new format means that the format should remain unchanged.
// To print out a new table, run: go test -run Formats -v.
var knownFormats = map[string]string{
	"*bytes.Buffer %s":                                "",
	"*cmd/compile/internal/gc.Mpflt %v":               "",
	"*cmd/compile/internal/gc.Mpint %v":               "",
	"*cmd/compile/internal/gc.Node %#v":               "",
	"*cmd/compile/internal/gc.Node %+S":               "",
	"*cmd/compile/internal/gc.Node %+v":               "",
	"*cmd/compile/internal/gc.Node %0j":               "",
	"*cmd/compile/internal/gc.Node %L":                "",
	"*cmd/compile/internal/gc.Node %S":                "",
	"*cmd/compile/internal/gc.Node %j":                "",
	"*cmd/compile/internal/gc.Node %p":                "",
	"*cmd/compile/internal/gc.Node %v":                "",
	"*cmd/compile/internal/ssa.Block %s":              "",
	"*cmd/compile/internal/ssa.Block %v":              "",
	"*cmd/compile/internal/ssa.Func %s":               "",
	"*cmd/compile/internal/ssa.SparseTreeNode %v":     "",
	"*cmd/compile/internal/ssa.Value %s":              "",
	"*cmd/compile/internal/ssa.Value %v":              "",
	"*cmd/compile/internal/ssa.sparseTreeMapEntry %v": "",
	"*cmd/compile/internal/types.Field %p":            "",
	"*cmd/compile/internal/types.Field %v":            "",
	"*cmd/compile/internal/types.Sym %+v":             "",
	"*cmd/compile/internal/types.Sym %-v":             "",
	"*cmd/compile/internal/types.Sym %0S":             "",
	"*cmd/compile/internal/types.Sym %S":              "",
	"*cmd/compile/internal/types.Sym %p":              "",
	"*cmd/compile/internal/types.Sym %v":              "",
	"*cmd/compile/internal/types.Type %#v":            "",
	"*cmd/compile/internal/types.Type %+v":            "",
	"*cmd/compile/internal/types.Type %-S":            "",
	"*cmd/compile/internal/types.Type %0S":            "",
	"*cmd/compile/internal/types.Type %L":             "",
	"*cmd/compile/internal/types.Type %S":             "",
	"*cmd/compile/internal/types.Type %p":             "",
	"*cmd/compile/internal/types.Type %s":             "",
	"*cmd/compile/internal/types.Type %v":             "",
	"*cmd/internal/obj.Addr %v":                       "",
	"*cmd/internal/obj.LSym %v":                       "",
	"*cmd/internal/obj.Prog %s":                       "",
	"*math/big.Int %#x":                               "",
	"*math/big.Int %s":                                "",
	"[16]byte %x":                                     "",
	"[]*cmd/compile/internal/gc.Node %v":              "",
	"[]*cmd/compile/internal/gc.Sig %#v":              "",
	"[]*cmd/compile/internal/ssa.Value %v":            "",
	"[]byte %s":                                       "",
	"[]byte %x":                                       "",
	"[]cmd/compile/internal/ssa.Edge %v":              "",
	"[]cmd/compile/internal/ssa.ID %v":                "",
	"[]string %v":                                     "",
	"bool %v":                                         "",
	"byte %08b":                                       "",
	"byte %c":                                         "",
	"cmd/compile/internal/arm.shift %d":               "",
	"cmd/compile/internal/gc.Class %d":                "",
	"cmd/compile/internal/gc.Ctype %d":                "",
	"cmd/compile/internal/gc.Ctype %v":                "",
	"cmd/compile/internal/gc.Level %d":                "",
	"cmd/compile/internal/gc.Level %v":                "",
	"cmd/compile/internal/gc.Nodes %#v":               "",
	"cmd/compile/internal/gc.Nodes %+v":               "",
	"cmd/compile/internal/gc.Nodes %.v":               "",
	"cmd/compile/internal/gc.Nodes %v":                "",
	"cmd/compile/internal/gc.Op %#v":                  "",
	"cmd/compile/internal/gc.Op %v":                   "",
	"cmd/compile/internal/gc.Val %#v":                 "",
	"cmd/compile/internal/gc.Val %T":                  "",
	"cmd/compile/internal/gc.Val %v":                  "",
	"cmd/compile/internal/gc.fmtMode %d":              "",
	"cmd/compile/internal/gc.initKind %d":             "",
	"cmd/compile/internal/ssa.BranchPrediction %d":    "",
	"cmd/compile/internal/ssa.Edge %v":                "",
	"cmd/compile/internal/ssa.GCNode %v":              "",
	"cmd/compile/internal/ssa.ID %d":                  "",
	"cmd/compile/internal/ssa.LocalSlot %v":           "",
	"cmd/compile/internal/ssa.Location %v":            "",
	"cmd/compile/internal/ssa.Op %s":                  "",
	"cmd/compile/internal/ssa.Op %v":                  "",
	"cmd/compile/internal/ssa.ValAndOff %s":           "",
	"cmd/compile/internal/ssa.rbrank %d":              "",
	"cmd/compile/internal/ssa.regMask %d":             "",
	"cmd/compile/internal/ssa.register %d":            "",
	"cmd/compile/internal/syntax.Expr %#v":            "",
	"cmd/compile/internal/syntax.Node %T":             "",
	"cmd/compile/internal/syntax.Operator %d":         "",
	"cmd/compile/internal/syntax.Operator %s":         "",
	"cmd/compile/internal/syntax.token %d":            "",
	"cmd/compile/internal/syntax.token %q":            "",
	"cmd/compile/internal/syntax.token %s":            "",
	"cmd/compile/internal/types.EType %d":             "",
	"cmd/compile/internal/types.EType %s":             "",
	"cmd/compile/internal/types.EType %v":             "",
	"cmd/internal/src.Pos %s":                         "",
	"cmd/internal/src.Pos %v":                         "",
	"error %v":                                        "",
	"float64 %.2f":                                    "",
	"float64 %.3f":                                    "",
	"float64 %.6g":                                    "",
	"float64 %g":                                      "",
	"int %-12d":                                       "",
	"int %-6d":                                        "",
	"int %-8o":                                        "",
	"int %6d":                                         "",
	"int %c":                                          "",
	"int %d":                                          "",
	"int %v":                                          "",
	"int %x":                                          "",
	"int16 %d":                                        "",
	"int16 %x":                                        "",
	"int32 %d":                                        "",
	"int32 %v":                                        "",
	"int32 %x":                                        "",
	"int64 %+d":                                       "",
	"int64 %-10d":                                     "",
	"int64 %X":                                        "",
	"int64 %d":                                        "",
	"int64 %v":                                        "",
	"int64 %x":                                        "",
	"int8 %d":                                         "",
	"int8 %x":                                         "",
	"interface{} %#v":                                 "",
	"interface{} %T":                                  "",
	"interface{} %q":                                  "",
	"interface{} %s":                                  "",
	"interface{} %v":                                  "",
	"map[*cmd/compile/internal/gc.Node]*cmd/compile/internal/ssa.Value %v": "",
	"reflect.Type %s":  "",
	"rune %#U":         "",
	"rune %c":          "",
	"string %-*s":      "",
	"string %-16s":     "",
	"string %.*s":      "",
	"string %q":        "",
	"string %s":        "",
	"string %v":        "",
	"time.Duration %d": "",
	"time.Duration %v": "",
	"uint %04x":        "",
	"uint %5d":         "",
	"uint %d":          "",
	"uint16 %d":        "",
	"uint16 %v":        "",
	"uint16 %x":        "",
	"uint32 %d":        "",
	"uint32 %x":        "",
	"uint64 %08x":      "",
	"uint64 %d":        "",
	"uint64 %x":        "",
	"uint8 %d":         "",
	"uint8 %x":         "",
	"uintptr %d":       "",
}
