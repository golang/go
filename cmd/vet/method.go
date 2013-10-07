// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains the code to check canonical methods.

package main

import (
	"fmt"
	"go/ast"
	"go/printer"
	"strings"
)

type MethodSig struct {
	args    []string
	results []string
}

// canonicalMethods lists the input and output types for Go methods
// that are checked using dynamic interface checks.  Because the
// checks are dynamic, such methods would not cause a compile error
// if they have the wrong signature: instead the dynamic check would
// fail, sometimes mysteriously.  If a method is found with a name listed
// here but not the input/output types listed here, vet complains.
//
// A few of the canonical methods have very common names.
// For example, a type might implement a Scan method that
// has nothing to do with fmt.Scanner, but we still want to check
// the methods that are intended to implement fmt.Scanner.
// To do that, the arguments that have a = prefix are treated as
// signals that the canonical meaning is intended: if a Scan
// method doesn't have a fmt.ScanState as its first argument,
// we let it go.  But if it does have a fmt.ScanState, then the
// rest has to match.
var canonicalMethods = map[string]MethodSig{
	// "Flush": {{}, {"error"}}, // http.Flusher and jpeg.writer conflict
	"Format":        {[]string{"=fmt.State", "rune"}, []string{}},                      // fmt.Formatter
	"GobDecode":     {[]string{"[]byte"}, []string{"error"}},                           // gob.GobDecoder
	"GobEncode":     {[]string{}, []string{"[]byte", "error"}},                         // gob.GobEncoder
	"MarshalJSON":   {[]string{}, []string{"[]byte", "error"}},                         // json.Marshaler
	"MarshalXML":    {[]string{"*xml.Encoder", "xml.StartElement"}, []string{"error"}}, // xml.Marshaler
	"Peek":          {[]string{"=int"}, []string{"[]byte", "error"}},                   // image.reader (matching bufio.Reader)
	"ReadByte":      {[]string{}, []string{"byte", "error"}},                           // io.ByteReader
	"ReadFrom":      {[]string{"=io.Reader"}, []string{"int64", "error"}},              // io.ReaderFrom
	"ReadRune":      {[]string{}, []string{"rune", "int", "error"}},                    // io.RuneReader
	"Scan":          {[]string{"=fmt.ScanState", "rune"}, []string{"error"}},           // fmt.Scanner
	"Seek":          {[]string{"=int64", "int"}, []string{"int64", "error"}},           // io.Seeker
	"UnmarshalJSON": {[]string{"[]byte"}, []string{"error"}},                           // json.Unmarshaler
	"UnmarshalXML":  {[]string{"*xml.Decoder", "xml.StartElement"}, []string{"error"}}, // xml.Unmarshaler
	"UnreadByte":    {[]string{}, []string{"error"}},
	"UnreadRune":    {[]string{}, []string{"error"}},
	"WriteByte":     {[]string{"byte"}, []string{"error"}},                // jpeg.writer (matching bufio.Writer)
	"WriteTo":       {[]string{"=io.Writer"}, []string{"int64", "error"}}, // io.WriterTo
}

func (f *File) checkCanonicalMethod(id *ast.Ident, t *ast.FuncType) {
	if !vet("methods") {
		return
	}
	// Expected input/output.
	expect, ok := canonicalMethods[id.Name]
	if !ok {
		return
	}

	// Actual input/output
	args := typeFlatten(t.Params.List)
	var results []ast.Expr
	if t.Results != nil {
		results = typeFlatten(t.Results.List)
	}

	// Do the =s (if any) all match?
	if !f.matchParams(expect.args, args, "=") || !f.matchParams(expect.results, results, "=") {
		return
	}

	// Everything must match.
	if !f.matchParams(expect.args, args, "") || !f.matchParams(expect.results, results, "") {
		expectFmt := id.Name + "(" + argjoin(expect.args) + ")"
		if len(expect.results) == 1 {
			expectFmt += " " + argjoin(expect.results)
		} else if len(expect.results) > 1 {
			expectFmt += " (" + argjoin(expect.results) + ")"
		}

		f.b.Reset()
		if err := printer.Fprint(&f.b, f.fset, t); err != nil {
			fmt.Fprintf(&f.b, "<%s>", err)
		}
		actual := f.b.String()
		actual = strings.TrimPrefix(actual, "func")
		actual = id.Name + actual

		f.Badf(id.Pos(), "method %s should have signature %s", actual, expectFmt)
	}
}

func argjoin(x []string) string {
	y := make([]string, len(x))
	for i, s := range x {
		if s[0] == '=' {
			s = s[1:]
		}
		y[i] = s
	}
	return strings.Join(y, ", ")
}

// Turn parameter list into slice of types
// (in the ast, types are Exprs).
// Have to handle f(int, bool) and f(x, y, z int)
// so not a simple 1-to-1 conversion.
func typeFlatten(l []*ast.Field) []ast.Expr {
	var t []ast.Expr
	for _, f := range l {
		if len(f.Names) == 0 {
			t = append(t, f.Type)
			continue
		}
		for _ = range f.Names {
			t = append(t, f.Type)
		}
	}
	return t
}

// Does each type in expect with the given prefix match the corresponding type in actual?
func (f *File) matchParams(expect []string, actual []ast.Expr, prefix string) bool {
	for i, x := range expect {
		if !strings.HasPrefix(x, prefix) {
			continue
		}
		if i >= len(actual) {
			return false
		}
		if !f.matchParamType(x, actual[i]) {
			return false
		}
	}
	if prefix == "" && len(actual) > len(expect) {
		return false
	}
	return true
}

// Does this one type match?
func (f *File) matchParamType(expect string, actual ast.Expr) bool {
	if strings.HasPrefix(expect, "=") {
		expect = expect[1:]
	}
	// Strip package name if we're in that package.
	if n := len(f.file.Name.Name); len(expect) > n && expect[:n] == f.file.Name.Name && expect[n] == '.' {
		expect = expect[n+1:]
	}

	// Overkill but easy.
	f.b.Reset()
	printer.Fprint(&f.b, f.fset, actual)
	return f.b.String() == expect
}
