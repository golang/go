// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains the exported entry points for invoking the parser.

package parser

import (
	"bytes"
	"go/ast"
	"go/scanner"
	"go/token"
	"io"
	"io/ioutil"
	"os"
	pathutil "path"
)


// If src != nil, readSource converts src to a []byte if possible;
// otherwise it returns an error. If src == nil, readSource returns
// the result of reading the file specified by filename.
//
func readSource(filename string, src interface{}) ([]byte, os.Error) {
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
			_, err := io.Copy(&buf, s)
			if err != nil {
				return nil, err
			}
			return buf.Bytes(), nil
		default:
			return nil, os.ErrorString("invalid source")
		}
	}

	return ioutil.ReadFile(filename)
}


func (p *parser) parseEOF() os.Error {
	p.expect(token.EOF)
	return p.GetError(scanner.Sorted)
}


// ParseExpr parses a Go expression and returns the corresponding
// AST node. The filename, src, and scope arguments have the same interpretation
// as for ParseFile. If there is an error, the result expression
// may be nil or contain a partial AST.
//
func ParseExpr(filename string, src interface{}, scope *ast.Scope) (ast.Expr, os.Error) {
	data, err := readSource(filename, src)
	if err != nil {
		return nil, err
	}

	var p parser
	p.init(filename, data, scope, 0)
	return p.parseExpr(), p.parseEOF()
}


// ParseStmtList parses a list of Go statements and returns the list
// of corresponding AST nodes. The filename, src, and scope arguments have the same
// interpretation as for ParseFile. If there is an error, the node
// list may be nil or contain partial ASTs.
//
func ParseStmtList(filename string, src interface{}, scope *ast.Scope) ([]ast.Stmt, os.Error) {
	data, err := readSource(filename, src)
	if err != nil {
		return nil, err
	}

	var p parser
	p.init(filename, data, scope, 0)
	return p.parseStmtList(), p.parseEOF()
}


// ParseDeclList parses a list of Go declarations and returns the list
// of corresponding AST nodes.  The filename, src, and scope arguments have the same
// interpretation as for ParseFile. If there is an error, the node
// list may be nil or contain partial ASTs.
//
func ParseDeclList(filename string, src interface{}, scope *ast.Scope) ([]ast.Decl, os.Error) {
	data, err := readSource(filename, src)
	if err != nil {
		return nil, err
	}

	var p parser
	p.init(filename, data, scope, 0)
	return p.parseDeclList(), p.parseEOF()
}


// ParseFile parses a Go source file and returns a File node.
//
// If src != nil, ParseFile parses the file source from src. src may
// be provided in a variety of formats. At the moment the following types
// are supported: string, []byte, and io.Reader. In this case, filename is
// only used for source position information and error messages.
//
// If src == nil, ParseFile parses the file specified by filename.
//
// If scope != nil, it is the immediately surrounding scope for the file
// (the package scope) and it is used to lookup and declare identifiers.
// When parsing multiple files belonging to a package, the same scope should
// be provided to all files.
//
// The mode parameter controls the amount of source text parsed and other
// optional parser functionality.
//
// If the source couldn't be read, the returned AST is nil and the error
// indicates the specific failure. If the source was read but syntax
// errors were found, the result is a partial AST (with ast.BadX nodes
// representing the fragments of erroneous source code). Multiple errors
// are returned via a scanner.ErrorList which is sorted by file position.
//
func ParseFile(filename string, src interface{}, scope *ast.Scope, mode uint) (*ast.File, os.Error) {
	data, err := readSource(filename, src)
	if err != nil {
		return nil, err
	}

	var p parser
	p.init(filename, data, scope, mode)
	return p.parseFile(), p.GetError(scanner.NoMultiples) // parseFile() reads to EOF
}


// ParseFiles calls ParseFile for each file in the filenames list and returns
// a map of package name -> package AST with all the packages found. The mode
// bits are passed to ParseFile unchanged.
//
// Files with parse errors are ignored. In this case the map of packages may
// be incomplete (missing packages and/or incomplete packages) and the last
// error encountered is returned.
//
func ParseFiles(filenames []string, scope *ast.Scope, mode uint) (map[string]*ast.Package, os.Error) {
	pkgs := make(map[string]*ast.Package)
	var err os.Error
	for _, filename := range filenames {
		var src *ast.File
		src, err = ParseFile(filename, nil, scope, mode)
		if err == nil {
			name := src.Name.Name()
			pkg, found := pkgs[name]
			if !found {
				pkg = &ast.Package{name, scope, make(map[string]*ast.File)}
				pkgs[name] = pkg
			}
			pkg.Files[filename] = src
		}
	}

	return pkgs, err
}


// ParseDir calls ParseFile for the files in the directory specified by path and
// returns a map of package name -> package AST with all the packages found. If
// filter != nil, only the files with os.Dir entries passing through the filter
// are considered. The mode bits are passed to ParseFile unchanged.
//
// If the directory couldn't be read, a nil map and the respective error are
// returned. If a parse error occured, a non-nil but incomplete map and the
// error are returned.
//
func ParseDir(path string, filter func(*os.Dir) bool, mode uint) (map[string]*ast.Package, os.Error) {
	fd, err := os.Open(path, os.O_RDONLY, 0)
	if err != nil {
		return nil, err
	}
	defer fd.Close()

	list, err := fd.Readdir(-1)
	if err != nil {
		return nil, err
	}

	filenames := make([]string, len(list))
	n := 0
	for i := 0; i < len(list); i++ {
		d := &list[i]
		if filter == nil || filter(d) {
			filenames[n] = pathutil.Join(path, d.Name)
			n++
		}
	}
	filenames = filenames[0:n]

	var scope *ast.Scope = nil // for now tracking of declarations is disabled
	return ParseFiles(filenames, scope, mode)
}
