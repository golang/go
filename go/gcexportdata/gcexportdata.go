// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package gcexportdata provides functions for locating, reading, and
// writing export data files containing type information produced by the
// gc compiler.  This package supports go1.7 export data format and all
// later versions.
//
// Although it might seem convenient for this package to live alongside
// go/types in the standard library, this would cause version skew
// problems for developer tools that use it, since they must be able to
// consume the outputs of the gc compiler both before and after a Go
// update such as from Go 1.7 to Go 1.8.  Because this package lives in
// golang.org/x/tools, sites can update their version of this repo some
// time before the Go 1.8 release and rebuild and redeploy their
// developer tools, which will then be able to consume both Go 1.7 and
// Go 1.8 export data files, so they will work before and after the
// Go update. (See discussion at https://golang.org/issue/15651.)
package gcexportdata // import "golang.org/x/tools/go/gcexportdata"

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"go/token"
	"go/types"
	"io"
	"os/exec"

	"golang.org/x/tools/internal/gcimporter"
)

// Find returns the name of an object (.o) or archive (.a) file
// containing type information for the specified import path,
// using the go command.
// If no file was found, an empty filename is returned.
//
// A relative srcDir is interpreted relative to the current working directory.
//
// Find also returns the package's resolved (canonical) import path,
// reflecting the effects of srcDir and vendoring on importPath.
//
// Deprecated: Use the higher-level API in golang.org/x/tools/go/packages,
// which is more efficient.
func Find(importPath, srcDir string) (filename, path string) {
	cmd := exec.Command("go", "list", "-json", "-export", "--", importPath)
	cmd.Dir = srcDir
	out, err := cmd.CombinedOutput()
	if err != nil {
		return "", ""
	}
	var data struct {
		ImportPath string
		Export     string
	}
	json.Unmarshal(out, &data)
	return data.Export, data.ImportPath
}

// NewReader returns a reader for the export data section of an object
// (.o) or archive (.a) file read from r.  The new reader may provide
// additional trailing data beyond the end of the export data.
func NewReader(r io.Reader) (io.Reader, error) {
	buf := bufio.NewReader(r)
	_, size, err := gcimporter.FindExportData(buf)
	if err != nil {
		return nil, err
	}

	if size >= 0 {
		// We were given an archive and found the __.PKGDEF in it.
		// This tells us the size of the export data, and we don't
		// need to return the entire file.
		return &io.LimitedReader{
			R: buf,
			N: size,
		}, nil
	} else {
		// We were given an object file. As such, we don't know how large
		// the export data is and must return the entire file.
		return buf, nil
	}
}

// readAll works the same way as io.ReadAll, but avoids allocations and copies
// by preallocating a byte slice of the necessary size if the size is known up
// front. This is always possible when the input is an archive. In that case,
// NewReader will return the known size using an io.LimitedReader.
func readAll(r io.Reader) ([]byte, error) {
	if lr, ok := r.(*io.LimitedReader); ok {
		data := make([]byte, lr.N)
		_, err := io.ReadFull(lr, data)
		return data, err
	}
	return io.ReadAll(r)
}

// Read reads export data from in, decodes it, and returns type
// information for the package.
//
// The package path (effectively its linker symbol prefix) is
// specified by path, since unlike the package name, this information
// may not be recorded in the export data.
//
// File position information is added to fset.
//
// Read may inspect and add to the imports map to ensure that references
// within the export data to other packages are consistent.  The caller
// must ensure that imports[path] does not exist, or exists but is
// incomplete (see types.Package.Complete), and Read inserts the
// resulting package into this map entry.
//
// On return, the state of the reader is undefined.
func Read(in io.Reader, fset *token.FileSet, imports map[string]*types.Package, path string) (*types.Package, error) {
	data, err := readAll(in)
	if err != nil {
		return nil, fmt.Errorf("reading export data for %q: %v", path, err)
	}

	if bytes.HasPrefix(data, []byte("!<arch>")) {
		return nil, fmt.Errorf("can't read export data for %q directly from an archive file (call gcexportdata.NewReader first to extract export data)", path)
	}

	// The indexed export format starts with an 'i'; the older
	// binary export format starts with a 'c', 'd', or 'v'
	// (from "version"). Select appropriate importer.
	if len(data) > 0 {
		switch data[0] {
		case 'v', 'c', 'd': // binary, till go1.10
			return nil, fmt.Errorf("binary (%c) import format is no longer supported", data[0])

		case 'i': // indexed, till go1.19
			_, pkg, err := gcimporter.IImportData(fset, imports, data[1:], path)
			return pkg, err

		case 'u': // unified, from go1.20
			_, pkg, err := gcimporter.UImportData(fset, imports, data[1:], path)
			return pkg, err

		default:
			l := len(data)
			if l > 10 {
				l = 10
			}
			return nil, fmt.Errorf("unexpected export data with prefix %q for path %s", string(data[:l]), path)
		}
	}
	return nil, fmt.Errorf("empty export data for %s", path)
}

// Write writes encoded type information for the specified package to out.
// The FileSet provides file position information for named objects.
func Write(out io.Writer, fset *token.FileSet, pkg *types.Package) error {
	if _, err := io.WriteString(out, "i"); err != nil {
		return err
	}
	return gcimporter.IExportData(out, fset, pkg)
}

// ReadBundle reads an export bundle from in, decodes it, and returns type
// information for the packages.
// File position information is added to fset.
//
// ReadBundle may inspect and add to the imports map to ensure that references
// within the export bundle to other packages are consistent.
//
// On return, the state of the reader is undefined.
//
// Experimental: This API is experimental and may change in the future.
func ReadBundle(in io.Reader, fset *token.FileSet, imports map[string]*types.Package) ([]*types.Package, error) {
	data, err := readAll(in)
	if err != nil {
		return nil, fmt.Errorf("reading export bundle: %v", err)
	}
	return gcimporter.IImportBundle(fset, imports, data)
}

// WriteBundle writes encoded type information for the specified packages to out.
// The FileSet provides file position information for named objects.
//
// Experimental: This API is experimental and may change in the future.
func WriteBundle(out io.Writer, fset *token.FileSet, pkgs []*types.Package) error {
	return gcimporter.IExportBundle(out, fset, pkgs)
}
