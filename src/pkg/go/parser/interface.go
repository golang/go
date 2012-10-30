// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains the exported entry points for invoking the parser.

package parser

import (
	"bytes"
	"errors"
	"go/ast"
	"go/token"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
)

// If src != nil, readSource converts src to a []byte if possible;
// otherwise it returns an error. If src == nil, readSource returns
// the result of reading the file specified by filename.
//
func readSource(filename string, src interface{}) ([]byte, error) {
	if src != nil {
		switch s := src.(type) {
		case string:
			return []byte(s), nil
		case []byte:
			return s, nil
		case *bytes.Buffer:
			// is io.Reader, but src is already available in []byte form
			if s != nil {
				return s.Bytes(), nil
			}
		case io.Reader:
			var buf bytes.Buffer
			if _, err := io.Copy(&buf, s); err != nil {
				return nil, err
			}
			return buf.Bytes(), nil
		}
		return nil, errors.New("invalid source")
	}
	return ioutil.ReadFile(filename)
}

// A Mode value is a set of flags (or 0).
// They control the amount of source code parsed and other optional
// parser functionality.
//
type Mode uint

const (
	PackageClauseOnly Mode = 1 << iota // parsing stops after package clause
	ImportsOnly                        // parsing stops after import declarations
	ParseComments                      // parse comments and add them to AST
	Trace                              // print a trace of parsed productions
	DeclarationErrors                  // report declaration errors
	SpuriousErrors                     // report all (not just the first) errors per line
)

// ParseFile parses the source code of a single Go source file and returns
// the corresponding ast.File node. The source code may be provided via
// the filename of the source file, or via the src parameter.
//
// If src != nil, ParseFile parses the source from src and the filename is
// only used when recording position information. The type of the argument
// for the src parameter must be string, []byte, or io.Reader.
// If src == nil, ParseFile parses the file specified by filename.
//
// The mode parameter controls the amount of source text parsed and other
// optional parser functionality. Position information is recorded in the
// file set fset.
//
// If the source couldn't be read, the returned AST is nil and the error
// indicates the specific failure. If the source was read but syntax
// errors were found, the result is a partial AST (with ast.Bad* nodes
// representing the fragments of erroneous source code). Multiple errors
// are returned via a scanner.ErrorList which is sorted by file position.
//
func ParseFile(fset *token.FileSet, filename string, src interface{}, mode Mode) (*ast.File, error) {
	// get source
	text, err := readSource(filename, src)
	if err != nil {
		return nil, err
	}

	// parse source
	var p parser
	p.init(fset, filename, text, mode)
	f := p.parseFile()
	if f == nil {
		// source is not a valid Go source file - satisfy
		// ParseFile API and return a valid (but) empty
		// *ast.File
		f = &ast.File{
			Name:  new(ast.Ident),
			Scope: ast.NewScope(nil),
		}
	}

	// sort errors
	if p.mode&SpuriousErrors == 0 {
		p.errors.RemoveMultiples()
	} else {
		p.errors.Sort()
	}

	return f, p.errors.Err()
}

// ParseDir calls ParseFile for the files in the directory specified by path and
// returns a map of package name -> package AST with all the packages found. If
// filter != nil, only the files with os.FileInfo entries passing through the filter
// are considered. The mode bits are passed to ParseFile unchanged. Position
// information is recorded in the file set fset.
//
// If the directory couldn't be read, a nil map and the respective error are
// returned. If a parse error occurred, a non-nil but incomplete map and the
// first error encountered are returned.
//
func ParseDir(fset *token.FileSet, path string, filter func(os.FileInfo) bool, mode Mode) (pkgs map[string]*ast.Package, first error) {
	fd, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer fd.Close()

	list, err := fd.Readdir(-1)
	if err != nil {
		return nil, err
	}

	pkgs = make(map[string]*ast.Package)
	for _, d := range list {
		if filter == nil || filter(d) {
			filename := filepath.Join(path, d.Name())
			if src, err := ParseFile(fset, filename, nil, mode); err == nil {
				name := src.Name.Name
				pkg, found := pkgs[name]
				if !found {
					pkg = &ast.Package{
						Name:  name,
						Files: make(map[string]*ast.File),
					}
					pkgs[name] = pkg
				}
				pkg.Files[filename] = src
			} else if first == nil {
				first = err
			}
		}
	}

	return
}

// ParseExpr is a convenience function for obtaining the AST of an expression x.
// The position information recorded in the AST is undefined.
//
func ParseExpr(x string) (ast.Expr, error) {
	// parse x within the context of a complete package for correct scopes;
	// use //line directive for correct positions in error messages and put
	// x alone on a separate line (handles line comments), followed by a ';'
	// to force an error if the expression is incomplete
	file, err := ParseFile(token.NewFileSet(), "", "package p;func _(){_=\n//line :1\n"+x+"\n;}", 0)
	if err != nil {
		return nil, err
	}
	return file.Decls[0].(*ast.FuncDecl).Body.List[0].(*ast.AssignStmt).Rhs[0], nil
}
