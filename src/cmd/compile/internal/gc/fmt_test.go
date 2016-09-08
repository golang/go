// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements TestFormats; a test that verifies
// all format strings in the compiler.
//
// TestFormats finds potential (Printf, etc.) format strings.
// If they are used in a call, the format verbs are verified
// based on the matching argument type against a precomputed
// table of valid formats (formatMapping, below). The table
// can be used to automatically rewrite format strings with
// the -u flag.
//
// Run as: go test -run Formats [-u]
//
// A formatMapping based on the existing formats is printed
// when the test is run in verbose mode (-v flag). The table
// needs to be updated whenever a new (type, format) combination
// is found and the format verb is not 'v' (as in "%v").
//
// Known bugs:
// - indexed format strings ("%[2]s", etc.) are not supported
//   (the test will fail)
// - format strings that are not simple string literals cannot
//   be updated automatically
//   (the test will fail with respective warnings)
//
package gc_test

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
	"sort"
	"strconv"
	"strings"
	"testing"
	"unicode/utf8"
)

var update = flag.Bool("u", false, "update format strings")

var fset = token.NewFileSet() // file set for all processed files
var typedPkg *types.Package   // the package we are type-checking

// A CallSite describes a function call that appears to contain
// a format string.
type CallSite struct {
	file  int           // buildPkg.GoFiles index
	call  *ast.CallExpr // call containing the format string
	arg   ast.Expr      // format argument (string literal or constant)
	str   string        // unquoted format string
	types []string      // argument types
}

func TestFormats(t *testing.T) {
	testenv.MustHaveGoBuild(t) // more restrictive than necessary, but that's ok

	// determine .go files
	buildPkg, err := build.ImportDir(".", 0)
	if err != nil {
		t.Fatal(err)
	}

	// parse all files
	files := make([]*ast.File, len(buildPkg.GoFiles))
	for i, filename := range buildPkg.GoFiles {
		f, err := parser.ParseFile(fset, filename, nil, parser.ParseComments)
		if err != nil {
			t.Fatal(err)
		}
		files[i] = f
	}

	// typecheck package
	conf := types.Config{Importer: importer.Default()}
	etypes := make(map[ast.Expr]types.TypeAndValue)
	typedPkg, err = conf.Check("some_path", fset, files, &types.Info{Types: etypes})
	if err != nil {
		t.Fatal(err)
	}

	// collect all potential format strings (for extra verification later)
	potentialFormats := make(map[*ast.BasicLit]bool)
	for _, file := range files {
		ast.Inspect(file, func(n ast.Node) bool {
			if s, ok := stringLit(n); ok && isFormat(s) {
				potentialFormats[n.(*ast.BasicLit)] = true
			}
			return true
		})
	}

	// collect all formats/arguments of calls with format strings
	var callSites []*CallSite
	for index, file := range files {
		ast.Inspect(file, func(n ast.Node) bool {
			if call, ok := n.(*ast.CallExpr); ok {
				// ignore blackisted functions
				if functionBlacklisted[nodeString(call.Fun)] {
					return true
				}
				// look for an arguments that might be a format string
				for i, arg := range call.Args {
					if s, ok := stringVal(etypes[arg]); ok && isFormat(s) {
						// make sure we have enough arguments
						n := formatArgs(s)
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
						callSites = append(callSites, &CallSite{
							file:  index,
							call:  call,
							arg:   arg,
							str:   s,
							types: argTypes,
						})
						break // at most one format per argument list
					}
				}
			}
			return true
		})
	}

	// test and rewrite formats
	keys := make(map[string]bool)
	dirty := false                         // dirty means some files have been modified
	dirtyFiles := make([]bool, len(files)) // dirtyFiles[i] means file i has been modified
	for _, p := range callSites {
		// test current format literal and determine updated one
		out := formatReplace(p.str, func(index int, in string) string {
			if in == "*" {
				return in
			}
			typ := p.types[index]
			key := typ + " " + in // e.g., "*Node %n"
			keys[key] = true
			out, found := formatMapping[key]
			if !found && in != "%v" { // always accept simple "%v"
				t.Errorf("%s: unknown format %q for %s argument", posString(p.arg), in, typ)
			}
			if out == "" {
				out = in
			}
			return out
		})

		// replace existing format literal if it changed
		if out != p.str {
			if testing.Verbose() {
				fmt.Printf("%s:\n\t   %q\n\t=> %q\n", posString(p.arg), p.str, out)
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
				panic("internal error: matching argument not found")
			}
			// we cannot replace the argument if it's not a string literal for now
			// (e.g., it may be "foo" + "bar")
			lit, ok := p.arg.(*ast.BasicLit)
			if !ok {
				continue // we issue a warning about missed format strings at the end
			}
			new := *lit                    // make a copy
			new.Value = strconv.Quote(out) // this may introduce "-quotes where there were `-quotes
			p.call.Args[index] = &new
			dirty = true
			dirtyFiles[p.file] = true // mark file as changed
		}
	}

	// write dirty files back
	if dirty && *update {
		for i, d := range dirtyFiles {
			if !d {
				continue
			}
			filename := buildPkg.GoFiles[i]
			var buf bytes.Buffer
			if err := format.Node(&buf, fset, files[i]); err != nil {
				t.Errorf("WARNING: formatting %s failed: %v", filename, err)
				continue
			}
			if err := ioutil.WriteFile(filename, buf.Bytes(), 0x666); err != nil {
				t.Errorf("WARNING: writing %s failed: %v", filename, err)
				continue
			}
			fmt.Println(filename)
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

	// report all keys found
	if len(keys) > 0 && testing.Verbose() {
		var list []string
		for s := range keys {
			list = append(list, fmt.Sprintf("%q: \"\",", s))
		}
		fmt.Println("\nvar formatMapping = map[string]string{")
		printList(list)
		fmt.Println("}")
	}

	// all format literals must in the potentialFormats set (self-verification)
	for _, p := range callSites {
		if lit, ok := p.arg.(*ast.BasicLit); ok && lit.Kind == token.STRING {
			if potentialFormats[lit] {
				// ok
				delete(potentialFormats, lit)
			} else {
				// this should never happen
				panic(fmt.Sprintf("internal error: format string not found (%s)", posString(lit)))
			}
		}
	}

	// if we have any strings left, we may need to update them manually
	if len(potentialFormats) > 0 && dirty && *update {
		var list []string
		for lit := range potentialFormats {
			list = append(list, fmt.Sprintf("%s: %s", posString(lit), nodeString(lit)))
		}
		fmt.Println("\nWARNING: Potentially missed format strings")
		printList(list)
		t.Fail()
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

// typeString returns a string representation of t.
func typeString(t types.Type) string {
	return types.TypeString(t, func(pkg *types.Package) string {
		// don't qualify type names of the typed package
		if pkg == typedPkg {
			return ""
		}
		return pkg.Path()
	})
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
			// accept any char except for % as format flag
			if r == '%' {
				if i-i0 == 1 {
					continue // skip "%%"
				}
				log.Fatalf("incorrect format string: %s", s)
			}
			if r >= 0 {
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

// formatArgs counts the number of format specifiers in s.
func formatArgs(s string) int {
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

func init() {
	// verify that formatMapping entries are correctly formatted
	for key, val := range formatMapping {
		// key must be "typename format" (format may be "*")
		i := strings.Index(key, " ")
		if i < 0 || key[i+1:] != "*" && !oneFormat(key[i+1:]) {
			log.Fatalf("incorrect formatMapping key: %q", key)
		}
		// val must be "format" or ""
		if val != "" && !oneFormat(val) {
			log.Fatalf("incorrect formatMapping key: %q", key)
		}
	}
}

// functionBlacklisted is the set of functions which are known to
// have format-like arguments but which don't do any formatting
// and thus can be ignored.
var functionBlacklisted = map[string]bool{
	"len": true,
	"strings.ContainsRune": true,
}

// formatMapping entries are of the form "typename oldformat" -> "newformat".
// An absent entry means that the format is not recognized as correct in test
// mode.
// An empty new format means that the existing format should remain unchanged.
// To generate a new table, run: go test -run Formats -v.
var formatMapping = map[string]string{
	"*Bits %v":                           "",
	"*Field %p":                          "",
	"*Field %v":                          "",
	"*Mpflt %v":                          "",
	"*Mpint %v":                          "",
	"*Node %#v":                          "",
	"*Node %+1v":                         "",
	"*Node %+v":                          "",
	"*Node %0j":                          "",
	"*Node %1v":                          "",
	"*Node %2v":                          "",
	"*Node %j":                           "",
	"*Node %p":                           "",
	"*Node %s":                           "",
	"*Node %v":                           "",
	"*Sym % v":                           "",
	"*Sym %+v":                           "",
	"*Sym %-v":                           "",
	"*Sym %01v":                          "",
	"*Sym %1v":                           "",
	"*Sym %p":                            "",
	"*Sym %s":                            "",
	"*Sym %v":                            "",
	"*Type % -v":                         "",
	"*Type %#v":                          "",
	"*Type %+v":                          "",
	"*Type %- v":                         "",
	"*Type %-1v":                         "",
	"*Type %-v":                          "",
	"*Type %01v":                         "",
	"*Type %1v":                          "",
	"*Type %2v":                          "",
	"*Type %p":                           "",
	"*Type %s":                           "",
	"*Type %v":                           "",
	"*cmd/compile/internal/big.Int %#x":  "",
	"*cmd/compile/internal/ssa.Block %v": "",
	"*cmd/compile/internal/ssa.Func %s":  "",
	"*cmd/compile/internal/ssa.Value %s": "",
	"*cmd/compile/internal/ssa.Value %v": "",
	"*cmd/internal/obj.Addr %v":          "",
	"*cmd/internal/obj.Prog %p":          "",
	"*cmd/internal/obj.Prog %s":          "",
	"*cmd/internal/obj.Prog %v":          "",
	"Class %d":                           "",
	"Ctype %d":                           "",
	"Ctype %v":                           "",
	"EType %d":                           "",
	"EType %s":                           "",
	"EType %v":                           "",
	"Level %d":                           "",
	"Level %v":                           "",
	"Nodes %#s":                          "",
	"Nodes %#v":                          "",
	"Nodes %+v":                          "",
	"Nodes %.v":                          "",
	"Nodes %v":                           "",
	"Op %#v":                             "",
	"Op %d":                              "",
	"Op %s":                              "",
	"Op %v":                              "",
	"Val %#v":                            "",
	"Val %s":                             "",
	"Val %v":                             "",
	"[16]byte %x":                        "",
	"[]*Node %v":                         "",
	"[]byte %s":                          "",
	"[]byte %x":                          "",
	"bool %t":                            "",
	"bool %v":                            "",
	"byte %02x":                          "",
	"byte %c":                            "",
	"cmd/compile/internal/ssa.Location %v":         "",
	"cmd/compile/internal/ssa.Type %s":             "",
	"cmd/compile/internal/syntax.Expr %#v":         "",
	"error %v":                                     "",
	"float64 %.2f":                                 "",
	"float64 %.6g":                                 "",
	"float64 %g":                                   "",
	"fmt.Stringer %T":                              "",
	"initKind %d":                                  "",
	"int %-12d":                                    "",
	"int %-2d":                                     "",
	"int %-6d":                                     "",
	"int %-8o":                                     "",
	"int %2d":                                      "",
	"int %6d":                                      "",
	"int %c":                                       "",
	"int %d":                                       "",
	"int %v":                                       "",
	"int %x":                                       "",
	"int16 %2d":                                    "",
	"int16 %d":                                     "",
	"int32 %4d":                                    "",
	"int32 %5d":                                    "",
	"int32 %d":                                     "",
	"int32 %v":                                     "",
	"int64 %+d":                                    "",
	"int64 %-10d":                                  "",
	"int64 %d":                                     "",
	"int64 %v":                                     "",
	"int8 %d":                                      "",
	"interface{} %#v":                              "",
	"interface{} %T":                               "",
	"interface{} %v":                               "",
	"map[*Node]*cmd/compile/internal/ssa.Value %v": "",
	"reflect.Type %s":                              "",
	"rune %#U":                                     "",
	"rune %c":                                      "",
	"rune %d":                                      "",
	"string %-16s":                                 "",
	"string %.*s":                                  "",
	"string %q":                                    "",
	"string %s":                                    "",
	"string %v":                                    "",
	"time.Duration %d":                             "",
	"uint %.4d":                                    "",
	"uint %04x":                                    "",
	"uint16 %d":                                    "",
	"uint16 %v":                                    "",
	"uint32 %#x":                                   "",
	"uint32 %d":                                    "",
	"uint64 %#x":                                   "",
	"uint64 %08x":                                  "",
	"uint8 %d":                                     "",
}
