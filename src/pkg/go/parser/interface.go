// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains the exported entry points for invoking the parser.

package parser

import (
	"bytes";
	"fmt";
	"go/ast";
	"go/parser";
	"go/scanner";
	"io";
	"os";
	pathutil "path";
	"strings";
)


// If src != nil, readSource converts src to a []byte if possible;
// otherwise it returns an error. If src == nil, readSource returns
// the result of reading the file specified by filename.
//
func readSource(filename string, src interface{}) ([]byte, os.Error) {
	if src != nil {
		switch s := src.(type) {
		case string:
			return strings.Bytes(s), nil;
		case []byte:
			return s, nil;
		case *bytes.Buffer:
			// is io.Reader, but src is already available in []byte form
			if s != nil {
				return s.Data(), nil;
			}
		case io.Reader:
			var buf bytes.Buffer;
			n, err := io.Copy(s, &buf);
			if err != nil {
				return nil, err;
			}
			return buf.Data(), nil;
		default:
			return nil, os.ErrorString("invalid source");
		}
	}

	return io.ReadFile(filename);
}


// ParseExpr parses a Go expression and returns the corresponding
// AST node. The filename and src arguments have the same interpretation
// as for ParseFile. If there is an error, the result expression
// may be nil or contain a partial AST.
//
func ParseExpr(filename string, src interface{}) (ast.Expr, os.Error) {
	data, err := readSource(filename, src);
	if err != nil {
		return nil, err;
	}

	var p parser;
	p.init(filename, data, 0);
	x := p.parseExpr();  // TODO 6g bug - function call order in expr lists
	return x, p.GetError(scanner.Sorted);
}


// ParseStmtList parses a list of Go statements and returns the list
// of corresponding AST nodes. The filename and src arguments have the same
// interpretation as for ParseFile. If there is an error, the node
// list may be nil or contain partial ASTs.
//
func ParseStmtList(filename string, src interface{}) ([]ast.Stmt, os.Error) {
	data, err := readSource(filename, src);
	if err != nil {
		return nil, err;
	}

	var p parser;
	p.init(filename, data, 0);
	list := p.parseStmtList();  // TODO 6g bug - function call order in expr lists
	return list, p.GetError(scanner.Sorted);
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
// The mode parameter controls the amount of source text parsed and other
// optional parser functionality.
//
// If the source couldn't be read, the returned AST is nil and the error
// indicates the specific failure. If the source was read but syntax
// errors were found, the result is a partial AST (with ast.BadX nodes
// representing the fragments of erroneous source code). Multiple errors
// are returned via a scanner.ErrorList which is sorted by file position.
//
func ParseFile(filename string, src interface{}, mode uint) (*ast.File, os.Error) {
	data, err := readSource(filename, src);
	if err != nil {
		return nil, err;
	}

	var p parser;
	p.init(filename, data, mode);
	prog := p.parseFile();  // TODO 6g bug - function call order in expr lists
	return prog, p.GetError(scanner.NoMultiples);
}


// ParsePkgFile parses the file specified by filename and returns the
// corresponding AST. If the file cannot be read, has syntax errors, or
// does not belong to the package (i.e., pkgname != "" and the package
// name in the file doesn't match pkkname), an error is returned. Mode
// flags that control the amount of source text parsed are ignored.
//
func ParsePkgFile(pkgname, filename string, mode uint) (*ast.File, os.Error) {
	src, err := io.ReadFile(filename);
	if err != nil {
		return nil, err;
	}

	if pkgname != "" {
		prog, err := ParseFile(filename, src, PackageClauseOnly);
		if err != nil {
			return nil, err;
		}
		if prog.Name.Value != pkgname {
			return nil, os.NewError(fmt.Sprintf("multiple packages found: %s, %s", prog.Name.Value, pkgname));
		}
	}

	// ignore flags that control partial parsing
	return ParseFile(filename, src, mode &^ (PackageClauseOnly | ImportsOnly));
}


// ParsePackage parses all files in the directory specified by path and
// returns an AST representing the package found. The set of files may be
// restricted by providing a non-nil filter function; only the files with
// (path-local) filenames passing through the filter are considered. If
// zero or more then one package is found, an error is returned. Mode
// flags that control the amount of source text parsed are ignored.
//
func ParsePackage(path string, filter func(string) bool, mode uint) (*ast.Package, os.Error) {
	fd, err := os.Open(path, os.O_RDONLY, 0);
	if err != nil {
		return nil, err;
	}

	list, err := fd.Readdirnames(-1);
	if err != nil {
		return nil, err;
	}

	name := "";
	files := make(map[string]*ast.File);
	for _, filename := range list {
		if filter == nil || filter(filename) {
			src, err := ParsePkgFile(name, pathutil.Join(path, filename), mode);
			if err != nil {
				return nil, err;
			}
			files[filename] = src;
			if name == "" {
				name = src.Name.Value;
			}
		}
	}

	if len(files) == 0 {
		return nil, os.NewError(path + ": no package found");
	}

	return &ast.Package{name, path, files}, nil;
}
