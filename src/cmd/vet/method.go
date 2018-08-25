// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains the code to check canonical methods.

package main

import (
	"go/ast"
	"go/types"
	"strings"
)

func init() {
	register("methods",
		"check that canonically named methods are canonically defined",
		checkCanonicalMethod,
		funcDecl, interfaceType)
}

type MethodSig struct {
	args    []string
	results []string
}

// canonicalMethods lists the input and output types for Go methods
// that are checked using dynamic interface checks. Because the
// checks are dynamic, such methods would not cause a compile error
// if they have the wrong signature: instead the dynamic check would
// fail, sometimes mysteriously. If a method is found with a name listed
// here but not the input/output types listed here, vet complains.
//
// A few of the canonical methods have very common names.
// For example, a type might implement a Scan method that
// has nothing to do with fmt.Scanner, but we still want to check
// the methods that are intended to implement fmt.Scanner.
// To do that, the arguments that have a = prefix are treated as
// signals that the canonical meaning is intended: if a Scan
// method doesn't have a fmt.ScanState as its first argument,
// we let it go. But if it does have a fmt.ScanState, then the
// rest has to match.
var canonicalMethods = map[string]MethodSig{
	// "Flush": {{}, {"error"}}, // http.Flusher and jpeg.writer conflict
	"Format":        {[]string{"=fmt.State", "rune"}, []string{}},                      // fmt.Formatter
	"GobDecode":     {[]string{"[]byte"}, []string{"error"}},                           // gob.GobDecoder
	"GobEncode":     {[]string{}, []string{"[]byte", "error"}},                         // gob.GobEncoder
	"MarshalJSON":   {[]string{}, []string{"[]byte", "error"}},                         // json.Marshaler
	"MarshalXML":    {[]string{"*xml.Encoder", "xml.StartElement"}, []string{"error"}}, // xml.Marshaler
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

func checkCanonicalMethod(f *File, node ast.Node) {
	switch n := node.(type) {
	case *ast.FuncDecl:
		if n.Recv != nil {
			canonicalMethod(f, n.Name)
		}
	case *ast.InterfaceType:
		for _, field := range n.Methods.List {
			for _, id := range field.Names {
				canonicalMethod(f, id)
			}
		}
	}
}

func canonicalMethod(f *File, id *ast.Ident) {
	// Expected input/output.
	expect, ok := canonicalMethods[id.Name]
	if !ok {
		return
	}
	sign := f.pkg.defs[id].Type().(*types.Signature)
	args := sign.Params()
	results := sign.Results()

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

		actual := sign.String()
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

// Does each type in expect with the given prefix match the corresponding type in actual?
func (f *File) matchParams(expect []string, actual *types.Tuple, prefix string) bool {
	for i, x := range expect {
		if !strings.HasPrefix(x, prefix) {
			continue
		}
		if i >= actual.Len() {
			return false
		}
		if !f.matchParamType(x, actual.At(i).Type()) {
			return false
		}
	}
	if prefix == "" && actual.Len() > len(expect) {
		return false
	}
	return true
}

// Does this one type match?
func (f *File) matchParamType(expect string, actual types.Type) bool {
	expect = strings.TrimPrefix(expect, "=")
	// Strip package name if we're in that package.
	if n := len(f.file.Name.Name); len(expect) > n && expect[:n] == f.file.Name.Name && expect[n] == '.' {
		expect = expect[n+1:]
	}

	// Overkill but easy.
	return actual.String() == expect
}
