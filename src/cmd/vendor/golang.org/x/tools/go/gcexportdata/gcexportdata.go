// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package gcexportdata provides functions for writing and reading
// export data, a serialized representation of a [types.Package].
// It describes the API of a Go package, including the names,
// kinds, types, and locations of all exported declarations.
//
// Files may be written and read using the [Write] and [Read]
// functions.
// Alternatively, files may be produced by the "go list -export" command.
// This command runs the type checker on each specified Go package and
// writes export data for each package into a file named by the Export
// field of go list's -json output.
// The [Read] function may then be used to read those files.
//
// The export data format evolves with each Go release. As a matter of
// policy, [Read] supports reading files produced by only the last two
// Go releases plus tip; see https://go.dev/issue/68898.
//
// The name of this package reflects its origins at a time when
// cmd/compile was named gc, and [Read] parsed export data
// from .a archives produced by the Go compiler.
// However, starting with go1.28, the format of the files
// produced by "go list -export" will diverge from the format
// used by the standard Go compiler.
// Consequently, this package will eventually stop being
// capable of reading those files.
// In the meantime, when using [Read] on a .a file written by the
// compiler, be aware that export data is not at the start of the file.
// Before calling Read, one must use [NewReader] to locate the export
// data section of the file.
// For more information on the compiler's export data, see the
// "Export" section in the GOROOT/src/cmd/compile/README file.
//
// # Deprecations
//
// The [NewImporter], [Find], and [NewReader] functions are deprecated
// and should not be used in new code.
// The [WriteBundle] and [ReadBundle] functions are experimental, and
// there is an open proposal to deprecate them (https://go.dev/issue/69573).
package gcexportdata

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
	out, err := cmd.Output()
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
//
// Deprecated: This package will stop supporting the reading of export
// data from compiler-produced archive files in Go 1.29.
func NewReader(r io.Reader) (io.Reader, error) {
	buf := bufio.NewReader(r)
	size, err := gcimporter.FindExportData(buf)
	if err != nil {
		return nil, err
	}

	// We were given an archive and found the __.PKGDEF in it.
	// This tells us the size of the export data, and we don't
	// need to return the entire file.
	return &io.LimitedReader{
		R: buf,
		N: size,
	}, nil
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
// Read is capable of reading export data produced by [Write] at the
// same source code version, or by the last two Go releases (plus tip)
// of the standard Go compiler. Reading files from older compilers may
// produce an error.
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
		case 'v', 'c', 'd':
			// binary, produced by cmd/compile till go1.10
			return nil, fmt.Errorf("binary (%c) import format is no longer supported", data[0])

		case 'i':
			// indexed, produced by cmd/compile till go1.19,
			// and also by [Write].
			return gcimporter.IImportData(fset, imports, data[1:], path)

		case 'u':
			// unified, produced by cmd/compile since go1.20
			_, pkg, err := gcimporter.UImportData(fset, imports, data[1:], path)
			return pkg, err

		default:
			l := min(len(data), 10)
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
