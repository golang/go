// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements the Import function for tests to use gc-generated object files.

package importer

import (
	"bufio"
	"fmt"
	"internal/exportdata"
	"internal/pkgbits"
	"internal/saferio"
	"io"
	"os"
	"strings"

	"cmd/compile/internal/types2"
)

// Import imports a gc-generated package given its import path and srcDir, adds
// the corresponding package object to the packages map, and returns the object.
// The packages map must contain all packages already imported.
//
// This function should only be used in tests.
func Import(packages map[string]*types2.Package, path, srcDir string, lookup func(path string) (io.ReadCloser, error)) (pkg *types2.Package, err error) {
	var rc io.ReadCloser
	var id string
	if lookup != nil {
		// With custom lookup specified, assume that caller has
		// converted path to a canonical import path for use in the map.
		if path == "unsafe" {
			return types2.Unsafe, nil
		}
		id = path

		// No need to re-import if the package was imported completely before.
		if pkg = packages[id]; pkg != nil && pkg.Complete() {
			return
		}
		f, err := lookup(path)
		if err != nil {
			return nil, err
		}
		rc = f
	} else {
		var filename string
		filename, id, err = exportdata.FindPkg(path, srcDir)
		if filename == "" {
			if path == "unsafe" {
				return types2.Unsafe, nil
			}
			return nil, err
		}

		// no need to re-import if the package was imported completely before
		if pkg = packages[id]; pkg != nil && pkg.Complete() {
			return
		}

		// open file
		f, err := os.Open(filename)
		if err != nil {
			return nil, err
		}
		defer func() {
			if err != nil {
				// add file name to error
				err = fmt.Errorf("%s: %v", filename, err)
			}
		}()
		rc = f
	}
	defer rc.Close()

	buf := bufio.NewReader(rc)
	hdr, size, err := exportdata.FindExportData(buf)
	if err != nil {
		return
	}

	switch hdr {
	case "$$\n":
		err = fmt.Errorf("import %q: old textual export format no longer supported (recompile package)", path)

	case "$$B\n":
		var exportFormat byte
		if exportFormat, err = buf.ReadByte(); err != nil {
			return
		}
		size--

		// The unified export format starts with a 'u'; the indexed export
		// format starts with an 'i'; and the older binary export format
		// starts with a 'c', 'd', or 'v' (from "version"). Select
		// appropriate importer.
		switch exportFormat {
		case 'u':
			// exported strings may contain "\n$$\n" - search backwards
			var data []byte
			var r io.Reader = buf
			if size >= 0 {
				if data, err = saferio.ReadData(r, uint64(size)); err != nil {
					return
				}
			} else if data, err = io.ReadAll(r); err != nil {
				return
			}
			s := string(data)
			s = s[:strings.LastIndex(s, "\n$$\n")]

			input := pkgbits.NewPkgDecoder(id, s)
			pkg = ReadPackage(nil, packages, input)
		default:
			err = fmt.Errorf("import %q: binary export format %q is no longer supported (recompile package)", path, exportFormat)
		}

	default:
		err = fmt.Errorf("import %q: unknown export data header: %q", path, hdr)
	}

	return
}
