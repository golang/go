// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"fmt"
	"internal/logosfile"
	"path"
	"strings"

	"cmd/compile/internal/syntax"
)

// unsafeOutsideUnsafeFile is pedantic::unsafe_outside_unsafe_file (SPEC
// Sec 1.2): an import of unsafe in a file whose name ends in neither
// _unsafe.go nor _test.go. The pedantic group defaults to allow, so the
// warning prints only under --pedantic or a stricter logos.toml key.
// Language neutral.
var unsafeOutsideUnsafeFile = Rule{
	ID:    "pedantic::unsafe_outside_unsafe_file",
	Group: Pedantic,
	Go:    true,
	Check: checkUnsafeImport,
}

func checkUnsafeImport(pass *Pass) {
	for _, f := range pass.Files {
		if unsafeImportAllowed(f) {
			continue
		}
		for _, d := range f.DeclList {
			imp, ok := d.(*syntax.ImportDecl)
			if !ok || imp.Path == nil || imp.Path.Kind != syntax.StringLit {
				continue
			}
			importPath, ok := literalString(imp.Path)
			if !ok || importPath != "unsafe" {
				continue
			}
			pass.Report(Diagnostic{
				Pos:     imp.Pos(),
				End:     imp.Pos(),
				Message: fmt.Sprintf("package unsafe imported by %s; the file name must end in _unsafe.go", path.Base(f.Pos().RelFilename())),
			})
		}
	}
}

// unsafeImportAllowed reports whether f may import unsafe: its name ends in
// _unsafe.go or _test.go, with the loGo/s extension in place of .go.
func unsafeImportAllowed(f *syntax.File) bool {
	name := f.Pos().RelFilename()
	stem, _, ok := logosfile.Split(name)
	if !ok {
		stem = strings.TrimSuffix(name, ".go")
	}
	return strings.HasSuffix(stem, "_unsafe") || strings.HasSuffix(stem, "_test")
}

// literalString returns the value of a string literal.
func literalString(lit *syntax.BasicLit) (string, bool) {
	text := lit.Value
	if len(text) < 2 || text[0] != '"' || text[len(text)-1] != '"' {
		return "", false
	}
	return text[1 : len(text)-1], true
}
