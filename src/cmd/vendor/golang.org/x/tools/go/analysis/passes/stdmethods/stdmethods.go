// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package stdmethods

import (
	_ "embed"
	"go/ast"
	"go/types"
	"strings"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/internal/analysisinternal"
)

//go:embed doc.go
var doc string

var Analyzer = &analysis.Analyzer{
	Name:     "stdmethods",
	Doc:      analysisinternal.MustExtractDoc(doc, "stdmethods"),
	URL:      "https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/stdmethods",
	Requires: []*analysis.Analyzer{inspect.Analyzer},
	Run:      run,
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
var canonicalMethods = map[string]struct{ args, results []string }{
	"As": {[]string{"any"}, []string{"bool"}}, // errors.As
	// "Flush": {{}, {"error"}}, // http.Flusher and jpeg.writer conflict
	"Format":        {[]string{"=fmt.State", "rune"}, []string{}},                      // fmt.Formatter
	"GobDecode":     {[]string{"[]byte"}, []string{"error"}},                           // gob.GobDecoder
	"GobEncode":     {[]string{}, []string{"[]byte", "error"}},                         // gob.GobEncoder
	"Is":            {[]string{"error"}, []string{"bool"}},                             // errors.Is
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
	"Unwrap":        {[]string{}, []string{"error"}},                      // errors.Unwrap
	"WriteByte":     {[]string{"byte"}, []string{"error"}},                // jpeg.writer (matching bufio.Writer)
	"WriteTo":       {[]string{"=io.Writer"}, []string{"int64", "error"}}, // io.WriterTo
}

func run(pass *analysis.Pass) (any, error) {
	inspect := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)

	nodeFilter := []ast.Node{
		(*ast.FuncDecl)(nil),
		(*ast.InterfaceType)(nil),
	}
	inspect.Preorder(nodeFilter, func(n ast.Node) {
		switch n := n.(type) {
		case *ast.FuncDecl:
			if n.Recv != nil {
				canonicalMethod(pass, n.Name)
			}
		case *ast.InterfaceType:
			for _, field := range n.Methods.List {
				for _, id := range field.Names {
					canonicalMethod(pass, id)
				}
			}
		}
	})
	return nil, nil
}

func canonicalMethod(pass *analysis.Pass, id *ast.Ident) {
	// Expected input/output.
	expect, ok := canonicalMethods[id.Name]
	if !ok {
		return
	}

	// Actual input/output
	sign := pass.TypesInfo.Defs[id].Type().(*types.Signature)
	args := sign.Params()
	results := sign.Results()

	// Special case: WriteTo with more than one argument,
	// not trying at all to implement io.WriterTo,
	// comes up often enough to skip.
	if id.Name == "WriteTo" && args.Len() > 1 {
		return
	}

	// Special case: Is, As and Unwrap only apply when type
	// implements error.
	if id.Name == "Is" || id.Name == "As" || id.Name == "Unwrap" {
		if recv := sign.Recv(); recv == nil || !implementsError(recv.Type()) {
			return
		}
	}

	// Special case: Unwrap has two possible signatures.
	// Check for Unwrap() []error here.
	if id.Name == "Unwrap" {
		if args.Len() == 0 && results.Len() == 1 {
			t := typeString(results.At(0).Type())
			if t == "error" || t == "[]error" {
				return
			}
		}
		pass.ReportRangef(id, "method Unwrap() should have signature Unwrap() error or Unwrap() []error")
		return
	}

	// Do the =s (if any) all match?
	if !matchParams(pass, expect.args, args, "=") || !matchParams(pass, expect.results, results, "=") {
		return
	}

	// Everything must match.
	if !matchParams(pass, expect.args, args, "") || !matchParams(pass, expect.results, results, "") {
		expectFmt := id.Name + "(" + argjoin(expect.args) + ")"
		if len(expect.results) == 1 {
			expectFmt += " " + argjoin(expect.results)
		} else if len(expect.results) > 1 {
			expectFmt += " (" + argjoin(expect.results) + ")"
		}

		actual := typeString(sign)
		actual = strings.TrimPrefix(actual, "func")
		actual = id.Name + actual

		pass.ReportRangef(id, "method %s should have signature %s", actual, expectFmt)
	}
}

func typeString(typ types.Type) string {
	return types.TypeString(typ, (*types.Package).Name)
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
func matchParams(pass *analysis.Pass, expect []string, actual *types.Tuple, prefix string) bool {
	for i, x := range expect {
		if !strings.HasPrefix(x, prefix) {
			continue
		}
		if i >= actual.Len() {
			return false
		}
		if !matchParamType(x, actual.At(i).Type()) {
			return false
		}
	}
	if prefix == "" && actual.Len() > len(expect) {
		return false
	}
	return true
}

// Does this one type match?
func matchParamType(expect string, actual types.Type) bool {
	expect = strings.TrimPrefix(expect, "=")
	// Overkill but easy.
	t := typeString(actual)
	return t == expect ||
		(t == "any" || t == "interface{}") && (expect == "any" || expect == "interface{}")
}

var errorType = types.Universe.Lookup("error").Type().Underlying().(*types.Interface)

func implementsError(actual types.Type) bool {
	return types.Implements(actual, errorType)
}
