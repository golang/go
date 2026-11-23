// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file locates the export data within compiler-produced archives.
// It should be kept in sync with $GOROOT/src/internal/exportdata/exportdata.go.

package gcexportdata

import (
	"bufio"
	"fmt"
	"strings"
	// This package is dependency-restricted; see x/tools/go/gcexportdata.TestDeps.
)

// findExportData positions the reader r at the beginning of the
// export data section of an underlying cmd/compile created archive
// file by reading from it. The reader must be positioned at the
// start of the file before calling this function.
// This returns the length of the export data in bytes.
//
// This function is needed by [gcexportdata.Read], which must
// accept inputs produced by the last two releases of cmd/compile,
// plus tip.
func findExportData(r *bufio.Reader) (size int64, err error) {
	arsize, err := findPackageDefinition(r)
	if err != nil {
		return
	}
	size = int64(arsize)

	objapi, headers, err := readObjectHeaders(r)
	if err != nil {
		return
	}
	size -= int64(len(objapi))
	for _, h := range headers {
		size -= int64(len(h))
	}

	// Check for the binary export data section header "$$B\n".
	line, err := r.ReadSlice('\n')
	if err != nil {
		return
	}
	hdr := string(line)
	if hdr != "$$B\n" {
		err = fmt.Errorf("unknown export data header: %q", hdr)
		return
	}
	size -= int64(len(hdr))

	// For files with a binary export data header "$$B\n",
	// these are always terminated by an end-of-section marker "\n$$\n".
	// So the last bytes must always be this constant.
	//
	// The end-of-section marker is not a part of the export data itself.
	// Do not include these in size.
	//
	// It would be nice to have sanity check that the final bytes after
	// the export data are indeed the end-of-section marker. The split
	// of gcexportdata.NewReader and gcexportdata.Read make checking this
	// ugly so this package gives up enforcing this. The compiler and go/types
	// importer do enforce this, which seems good enough.
	const endofsection = "\n$$\n"
	size -= int64(len(endofsection))

	if size < 0 {
		err = fmt.Errorf("invalid size (%d) in the archive file: %d bytes remain without section headers (recompile package)", arsize, size)
		return
	}

	return
}

// findPackageDefinition positions the reader r at the beginning of a package
// definition file ("__.PKGDEF") within a GC-created archive by reading
// from it, and returns the size of the package definition file in the archive.
//
// The reader must be positioned at the start of the archive file before calling
// this function, and "__.PKGDEF" is assumed to be the first file in the archive.
//
// See cmd/internal/archive for details on the archive format.
func findPackageDefinition(r *bufio.Reader) (size int, err error) {
	// Uses ReadSlice to limit risk of malformed inputs.

	// Read first line to make sure this is an object file.
	line, err := r.ReadSlice('\n')
	if err != nil {
		err = fmt.Errorf("can't find export data (%v)", err)
		return
	}

	// Is the first line an archive file signature?
	if string(line) != "!<arch>\n" {
		err = fmt.Errorf("not the start of an archive file (%q)", line)
		return
	}

	// package export block should be first
	size = readArchiveHeader(r, "__.PKGDEF")
	if size <= 0 {
		err = fmt.Errorf("not a package file")
		return
	}

	return
}

// readObjectHeaders reads object headers from the reader. Object headers are
// lines that do not start with an end-of-section marker "$$". The first header
// is the objabi header. On success, the reader will be positioned at the beginning
// of the end-of-section marker.
//
// It returns an error if any header does not fit in r.Size() bytes.
func readObjectHeaders(r *bufio.Reader) (objapi string, headers []string, err error) {
	// line is a temporary buffer for headers.
	// Use bounded reads (ReadSlice, Peek) to limit risk of malformed inputs.
	var line []byte

	// objapi header should be the first line
	if line, err = r.ReadSlice('\n'); err != nil {
		err = fmt.Errorf("can't find export data (%v)", err)
		return
	}
	objapi = string(line)

	// objapi header begins with "go object ".
	if !strings.HasPrefix(objapi, "go object ") {
		err = fmt.Errorf("not a go object file: %s", objapi)
		return
	}

	// process remaining object header lines
	for {
		// check for an end of section marker "$$"
		line, err = r.Peek(2)
		if err != nil {
			return
		}
		if string(line) == "$$" {
			return // stop
		}

		// read next header
		line, err = r.ReadSlice('\n')
		if err != nil {
			return
		}
		headers = append(headers, string(line))
	}
}
