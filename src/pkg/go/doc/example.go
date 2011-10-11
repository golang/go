// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Extract example functions from package ASTs.

package doc

import (
	"go/ast"
	"strings"
	"unicode"
	"utf8"
)

type Example struct {
	Name   string         // name of the item being demonstrated
	Body   *ast.BlockStmt // code
	Output string         // expected output
}

func Examples(pkg *ast.Package) []*Example {
	var examples []*Example
	for _, src := range pkg.Files {
		for _, decl := range src.Decls {
			f, ok := decl.(*ast.FuncDecl)
			if !ok {
				continue
			}
			name := f.Name.Name
			if !isTest(name, "Example") {
				continue
			}
			examples = append(examples, &Example{
				Name:   name[len("Example"):],
				Body:   f.Body,
				Output: CommentText(f.Doc),
			})
		}
	}
	return examples
}

// isTest tells whether name looks like a test, example, or benchmark.
// It is a Test (say) if there is a character after Test that is not a
// lower-case letter. (We don't want Testiness.)
func isTest(name, prefix string) bool {
	if !strings.HasPrefix(name, prefix) {
		return false
	}
	if len(name) == len(prefix) { // "Test" is ok
		return true
	}
	rune, _ := utf8.DecodeRuneInString(name[len(prefix):])
	return !unicode.IsLower(rune)
}
