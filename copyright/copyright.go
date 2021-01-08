// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package copyright checks that files have the correct copyright notices.
package copyright

import (
	"go/ast"
	"go/parser"
	"go/token"
	"io/ioutil"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

func checkCopyright(dir string) ([]string, error) {
	var files []string
	err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			// Skip directories like ".git".
			if strings.HasPrefix(info.Name(), ".") {
				return filepath.SkipDir
			}
			return nil
		}
		needsCopyright, err := checkFile(dir, path)
		if err != nil {
			return err
		}
		if needsCopyright {
			files = append(files, path)
		}
		return nil
	})
	return files, err
}

var copyrightRe = regexp.MustCompile(`Copyright \d{4} The Go Authors. All rights reserved.
Use of this source code is governed by a BSD-style
license that can be found in the LICENSE file.`)

func checkFile(toolsDir, filename string) (bool, error) {
	// Only check Go files.
	if !strings.HasSuffix(filename, "go") {
		return false, nil
	}
	// Don't check testdata files.
	normalized := strings.TrimPrefix(filepath.ToSlash(filename), filepath.ToSlash(toolsDir))
	if strings.Contains(normalized, "/testdata/") {
		return false, nil
	}
	// goyacc is the only file with a different copyright header.
	if strings.HasSuffix(normalized, "cmd/goyacc/yacc.go") {
		return false, nil
	}
	content, err := ioutil.ReadFile(filename)
	if err != nil {
		return false, err
	}
	fset := token.NewFileSet()
	parsed, err := parser.ParseFile(fset, filename, content, parser.ParseComments)
	if err != nil {
		return false, err
	}
	// Don't require headers on generated files.
	if isGenerated(fset, parsed) {
		return false, nil
	}
	shouldAddCopyright := true
	for _, c := range parsed.Comments {
		// The copyright should appear before the package declaration.
		if c.Pos() > parsed.Package {
			break
		}
		if copyrightRe.MatchString(c.Text()) {
			shouldAddCopyright = false
			break
		}
	}
	return shouldAddCopyright, nil
}

// Copied from golang.org/x/tools/internal/lsp/source/util.go.
// Matches cgo generated comment as well as the proposed standard:
//	https://golang.org/s/generatedcode
var generatedRx = regexp.MustCompile(`// .*DO NOT EDIT\.?`)

func isGenerated(fset *token.FileSet, file *ast.File) bool {
	for _, commentGroup := range file.Comments {
		for _, comment := range commentGroup.List {
			if matched := generatedRx.MatchString(comment.Text); !matched {
				continue
			}
			// Check if comment is at the beginning of the line in source.
			if pos := fset.Position(comment.Slash); pos.Column == 1 {
				return true
			}
		}
	}
	return false
}
