// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package analysisinternal

import (
	"fmt"
	"go/parser"
	"go/token"
	"strings"
)

// MustExtractDoc is like [ExtractDoc] but it panics on error.
//
// To use, define a doc.go file such as:
//
//	// Package halting defines an analyzer of program termination.
//	//
//	// # Analyzer halting
//	//
//	// halting: reports whether execution will halt.
//	//
//	// The halting analyzer reports a diagnostic for functions
//	// that run forever. To suppress the diagnostics, try inserting
//	// a 'break' statement into each loop.
//	package halting
//
//	import _ "embed"
//
//	//go:embed doc.go
//	var doc string
//
// And declare your analyzer as:
//
//	var Analyzer = &analysis.Analyzer{
//		Name:             "halting",
//		Doc:              analysisutil.MustExtractDoc(doc, "halting"),
//		...
//	}
func MustExtractDoc(content, name string) string {
	doc, err := ExtractDoc(content, name)
	if err != nil {
		panic(err)
	}
	return doc
}

// ExtractDoc extracts a section of a package doc comment from the
// provided contents of an analyzer package's doc.go file.
//
// A section is a portion of the comment between one heading and
// the next, using this form:
//
//	# Analyzer NAME
//
//	NAME: SUMMARY
//
//	Full description...
//
// where NAME matches the name argument, and SUMMARY is a brief
// verb-phrase that describes the analyzer. The following lines, up
// until the next heading or the end of the comment, contain the full
// description. ExtractDoc returns the portion following the colon,
// which is the form expected by Analyzer.Doc.
//
// Example:
//
//	# Analyzer printf
//
//	printf: checks consistency of calls to printf
//
//	The printf analyzer checks consistency of calls to printf.
//	Here is the complete description...
//
// This notation allows a single doc comment to provide documentation
// for multiple analyzers, each in its own section.
// The HTML anchors generated for each heading are predictable.
//
// It returns an error if the content was not a valid Go source file
// containing a package doc comment with a heading of the required
// form.
//
// This machinery enables the package documentation (typically
// accessible via the web at https://pkg.go.dev/) and the command
// documentation (typically printed to a terminal) to be derived from
// the same source and formatted appropriately.
func ExtractDoc(content, name string) (string, error) {
	if content == "" {
		return "", fmt.Errorf("empty Go source file")
	}
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "", content, parser.ParseComments|parser.PackageClauseOnly)
	if err != nil {
		return "", fmt.Errorf("not a Go source file")
	}
	if f.Doc == nil {
		return "", fmt.Errorf("Go source file has no package doc comment")
	}
	for section := range strings.SplitSeq(f.Doc.Text(), "\n# ") {
		if body := strings.TrimPrefix(section, "Analyzer "+name); body != section &&
			body != "" &&
			body[0] == '\r' || body[0] == '\n' {
			body = strings.TrimSpace(body)
			rest := strings.TrimPrefix(body, name+":")
			if rest == body {
				return "", fmt.Errorf("'Analyzer %s' heading not followed by '%s: summary...' line", name, name)
			}
			return strings.TrimSpace(rest), nil
		}
	}
	return "", fmt.Errorf("package doc comment contains no 'Analyzer %s' heading", name)
}
