// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package gccgoexportdata provides functions for reading export data
// files containing type information produced by the gccgo compiler.
//
// This package is a stop-gap until such time as gccgo uses the same
// export data format as gc; see Go issue 17573. Once that occurs, this
// package will be deprecated and eventually deleted.
package gccgoexportdata

// TODO(adonovan): add Find, Write, Importer to the API,
// for symmetry with gcexportdata.

import (
	"bytes"
	"debug/elf"
	"fmt"
	"go/token"
	"go/types"
	"io"
	"io/ioutil"
	"strconv"
	"strings"

	"golang.org/x/tools/go/internal/gccgoimporter"
)

// CompilerInfo executes the specified gccgo compiler and returns
// information about it: its version (e.g. "4.8.0"), its target triple
// (e.g. "x86_64-unknown-linux-gnu"), and the list of directories it
// searches to find standard packages.
func CompilerInfo(gccgo string) (version, triple string, dirs []string, err error) {
	var inst gccgoimporter.GccgoInstallation
	err = inst.InitFromDriver(gccgo)
	if err == nil {
		version = inst.GccVersion
		triple = inst.TargetTriple
		dirs = inst.SearchPaths()
	}
	return
}

// NewReader returns a reader for the export data section of an object
// (.o) or archive (.a) file read from r.
func NewReader(r io.Reader) (io.Reader, error) {
	data, err := ioutil.ReadAll(r)
	if err != nil {
		return nil, err
	}

	// If the file is an archive, extract the first section.
	const archiveMagic = "!<arch>\n"
	if bytes.HasPrefix(data, []byte(archiveMagic)) {
		section, err := firstSection(data[len(archiveMagic):])
		if err != nil {
			return nil, err
		}
		data = section
	}

	// Data contains an ELF file with a .go_export section.
	// ELF magic number is "\x7fELF".
	ef, err := elf.NewFile(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	sec := ef.Section(".go_export")
	if sec == nil {
		return nil, fmt.Errorf("no .go_export section")
	}
	return sec.Open(), nil
}

// firstSection returns the contents of the first non-empty section of the archive file.
func firstSection(a []byte) ([]byte, error) {
	for len(a) >= 60 {
		var hdr []byte
		hdr, a = a[:60], a[60:]

		modeStr := string(string(hdr[40:48]))
		mode, err := strconv.Atoi(strings.TrimSpace(modeStr))
		if err != nil {
			return nil, fmt.Errorf("invalid mode: %q", modeStr)
		}

		sizeStr := string(hdr[48:58])
		size, err := strconv.Atoi(strings.TrimSpace(sizeStr))
		if err != nil {
			return nil, fmt.Errorf("invalid size: %q", sizeStr)
		}

		var payload []byte
		payload, a = a[:size], a[size:]

		if mode == 0 {
			continue // skip "/"
		}

		return payload, nil
	}
	return nil, fmt.Errorf("archive has no non-empty sections")
}

// Read reads export data from in, decodes it, and returns type
// information for the package.
// The package name is specified by path.
//
// The FileSet parameter is currently unused but exists for symmetry
// with gcexportdata.
//
// Read may inspect and add to the imports map to ensure that references
// within the export data to other packages are consistent.  The caller
// must ensure that imports[path] does not exist, or exists but is
// incomplete (see types.Package.Complete), and Read inserts the
// resulting package into this map entry.
//
// On return, the state of the reader is undefined.
func Read(in io.Reader, _ *token.FileSet, imports map[string]*types.Package, path string) (*types.Package, error) {
	return gccgoimporter.Parse(in, imports, path)
}
