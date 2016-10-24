package gccgoimporter

// This file opens a back door to the parser for golang.org/x/tools/go/gccgoexportdata.

import (
	"go/types"
	"io"
)

// Parse reads and parses gccgo export data from in and constructs a
// Package, inserting it into the imports map.
func Parse(in io.Reader, imports map[string]*types.Package, path string) (_ *types.Package, err error) {
	var p parser
	p.init(path, in, imports)
	defer func() {
		switch x := recover().(type) {
		case nil:
			// success
		case importError:
			err = x
		default:
			panic(x) // resume unexpected panic
		}
	}()
	pkg := p.parsePackage()
	imports[path] = pkg
	return pkg, err
}
