// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"sync"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/lsp/protocol"
)

type View struct {
	mu sync.Mutex // protects all mutable state of the view

	Config *packages.Config

	files map[protocol.DocumentURI]*File
}

func NewView() *View {
	return &View{
		Config: &packages.Config{
			Mode:    packages.LoadSyntax,
			Fset:    token.NewFileSet(),
			Tests:   true,
			Overlay: make(map[string][]byte),
		},
		files: make(map[protocol.DocumentURI]*File),
	}
}

// GetFile returns a File for the given uri.
// It will always succeed, adding the file to the managed set if needed.
func (v *View) GetFile(uri protocol.DocumentURI) *File {
	v.mu.Lock()
	f, found := v.files[uri]
	if !found {
		f = &File{
			URI:  uri,
			view: v,
		}
		v.files[f.URI] = f
	}
	v.mu.Unlock()
	return f
}

// TypeCheck type-checks the package for the given package path.
func (v *View) TypeCheck(uri protocol.DocumentURI) (*packages.Package, error) {
	v.mu.Lock()
	defer v.mu.Unlock()
	path, err := FromURI(uri)
	if err != nil {
		return nil, err
	}
	pkgs, err := packages.Load(v.Config, fmt.Sprintf("file=%s", path))
	if len(pkgs) == 0 {
		if err == nil {
			err = fmt.Errorf("no packages found for %s", path)
		}
		return nil, err
	}
	pkg := pkgs[0]
	return pkg, nil
}

func (v *View) TypeCheckAtPosition(uri protocol.DocumentURI, position protocol.Position) (*packages.Package, *ast.File, token.Pos, error) {
	v.mu.Lock()
	defer v.mu.Unlock()
	filename, err := FromURI(uri)
	if err != nil {
		return nil, nil, token.NoPos, err
	}

	var mu sync.Mutex
	var qfileContent []byte

	cfg := &packages.Config{
		Mode:       v.Config.Mode,
		Dir:        v.Config.Dir,
		Env:        v.Config.Env,
		BuildFlags: v.Config.BuildFlags,
		Fset:       v.Config.Fset,
		Tests:      v.Config.Tests,
		Overlay:    v.Config.Overlay,
		ParseFile: func(fset *token.FileSet, current string, data []byte) (*ast.File, error) {
			// Save the file contents for use later in determining the query position.
			if sameFile(current, filename) {
				mu.Lock()
				qfileContent = data
				mu.Unlock()
			}
			return parser.ParseFile(fset, current, data, parser.AllErrors)
		},
	}
	pkgs, err := packages.Load(cfg, fmt.Sprintf("file=%s", filename))
	if len(pkgs) == 0 {
		if err == nil {
			err = fmt.Errorf("no package found for %s", filename)
		}
		return nil, nil, token.NoPos, err
	}
	pkg := pkgs[0]

	var qpos token.Pos
	var qfile *ast.File
	for _, file := range pkg.Syntax {
		tokfile := pkg.Fset.File(file.Pos())
		if tokfile == nil || tokfile.Name() != filename {
			continue
		}
		pos := positionToPos(tokfile, qfileContent, int(position.Line), int(position.Character))
		if !pos.IsValid() {
			return nil, nil, token.NoPos, fmt.Errorf("invalid position for %s", filename)
		}
		qfile = file
		qpos = pos
		break
	}

	if qfile == nil || qpos == token.NoPos {
		return nil, nil, token.NoPos, fmt.Errorf("unable to find position %s:%v:%v", filename, position.Line, position.Character)
	}
	return pkg, qfile, qpos, nil
}

// trimAST clears any part of the AST not relevant to type checking
// expressions at pos.
func trimAST(file *ast.File, pos token.Pos) {
	ast.Inspect(file, func(n ast.Node) bool {
		if n == nil {
			return false
		}
		if pos < n.Pos() || pos >= n.End() {
			switch n := n.(type) {
			case *ast.FuncDecl:
				n.Body = nil
			case *ast.BlockStmt:
				n.List = nil
			case *ast.CaseClause:
				n.Body = nil
			case *ast.CommClause:
				n.Body = nil
			case *ast.CompositeLit:
				// Leave elts in place for [...]T
				// array literals, because they can
				// affect the expression's type.
				if !isEllipsisArray(n.Type) {
					n.Elts = nil
				}
			}
		}
		return true
	})
}

func isEllipsisArray(n ast.Expr) bool {
	at, ok := n.(*ast.ArrayType)
	if !ok {
		return false
	}
	_, ok = at.Len.(*ast.Ellipsis)
	return ok
}

func sameFile(filename1, filename2 string) bool {
	if filepath.Base(filename1) != filepath.Base(filename2) {
		return false
	}
	finfo1, err := os.Stat(filename1)
	if err != nil {
		return false
	}
	finfo2, err := os.Stat(filename2)
	if err != nil {
		return false
	}
	return os.SameFile(finfo1, finfo2)
}

// positionToPos converts a 0-based line and column number in a file
// to a token.Pos. It returns NoPos if the file did not contain the position.
func positionToPos(file *token.File, content []byte, line, col int) token.Pos {
	if file.Size() != len(content) {
		return token.NoPos
	}
	if file.LineCount() < int(line) { // these can be equal if the last line is empty
		return token.NoPos
	}
	start := 0
	for i := 0; i < int(line); i++ {
		if start >= len(content) {
			return token.NoPos
		}
		index := bytes.IndexByte(content[start:], '\n')
		if index == -1 {
			return token.NoPos
		}
		start += (index + 1)
	}
	offset := start + int(col)
	if offset > file.Size() {
		return token.NoPos
	}
	return file.Pos(offset)
}
