// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements FindExportData.

package importer

import (
	"bufio"
	"cmd/internal/archive"
	"fmt"
	"strings"
)

// FindExportData positions the reader r at the beginning of the
// export data section of an underlying GC-created object/archive
// file by reading from it. The reader must be positioned at the
// start of the file before calling this function. The hdr result
// is the string before the export data, either "$$" or "$$B".
//
// If size is non-negative, it's the number of bytes of export data
// still available to read from r.
//
// This function should only be used in tests.
func FindExportData(r *bufio.Reader) (hdr string, size int, err error) {
	// TODO(taking): Move into a src/internal package then
	// dedup with cmd/compile/internal/noder.findExportData and go/internal/gcimporter.FindExportData.

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
	size = archive.ReadHeader(r, "__.PKGDEF")
	if size <= 0 {
		err = fmt.Errorf("not a package file")
		return
	}

	// Read first line of __.PKGDEF data, so that line
	// is once again the first line of the input.
	if line, err = r.ReadSlice('\n'); err != nil {
		err = fmt.Errorf("can't find export data (%v)", err)
		return
	}

	// Now at __.PKGDEF in archive. line should begin with "go object ".
	if !strings.HasPrefix(string(line), "go object ") {
		err = fmt.Errorf("not a Go object file")
		return
	}
	size -= len(line)

	// Skip over object header to export data.
	// Begins after first line starting with $$.
	for line[0] != '$' {
		if line, err = r.ReadSlice('\n'); err != nil {
			err = fmt.Errorf("can't find export data (%v)", err)
			return
		}
		size -= len(line)
	}
	hdr = string(line)

	return
}
