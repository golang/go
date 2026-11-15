// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logosfile

import (
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"testing"
)

var aliasSpellings = []string{"📜", "⚡", "ʕ◔ϖ◔ʔ"}

var aliasCodePoints = []int64{0x1F4DC, 0x26A1}

func TestNoSecondExtensionList(t *testing.T) {
	src := filepath.Join(runtime.GOROOT(), "src")
	if _, err := os.Stat(src); err != nil {
		t.Skip("no source tree")
	}
	err := filepath.WalkDir(src, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			if d.Name() == "testdata" || d.Name() == "vendor" || path == filepath.Join(src, "internal", "logosfile") {
				return filepath.SkipDir
			}
			return nil
		}
		if strings.HasSuffix(path, ".go") && !strings.HasSuffix(path, "_test.go") {
			reportAliasLiterals(t, src, path)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}

func reportAliasLiterals(t *testing.T, src, path string) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, path, nil, parser.SkipObjectResolution)
	if err != nil {
		t.Error(err)
		return
	}
	rel, _ := filepath.Rel(src, path)
	ast.Inspect(f, func(n ast.Node) bool {
		if lit, ok := n.(*ast.BasicLit); ok && namesAlias(lit) {
			t.Errorf("%s:%d: second list of loGo/s extensions: %s", rel, fset.Position(lit.Pos()).Line, lit.Value)
		}
		return true
	})
}

func namesAlias(lit *ast.BasicLit) bool {
	switch lit.Kind {
	case token.INT:
		v, err := strconv.ParseInt(lit.Value, 0, 64)
		return err == nil && slices.Contains(aliasCodePoints, v)
	case token.STRING, token.CHAR:
		v, err := strconv.Unquote(lit.Value)
		if err != nil {
			return false
		}
		for _, s := range aliasSpellings {
			if strings.Contains(v, s) {
				return true
			}
		}
	}
	return false
}
