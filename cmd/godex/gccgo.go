// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements access to gccgo-generated export data.

package main

import (
	"debug/elf"
	"fmt"
	"io"
	"os"

	"code.google.com/p/go.tools/go/gccgoimporter"
	"code.google.com/p/go.tools/go/importer"
	"code.google.com/p/go.tools/go/types"
)

func init() {
	// importer for default gccgo
	var inst gccgoimporter.GccgoInstallation
	inst.InitFromDriver("gccgo")
	register("gccgo", inst.GetImporter(nil))

	// importer for gccgo using condensed export format (experimental)
	register("gccgo-new", gccgoNewImporter)
}

func gccgoNewImporter(packages map[string]*types.Package, path string) (*types.Package, error) {
	reader, closer, err := openGccgoExportFile(path)
	if err != nil {
		return nil, err
	}
	defer closer.Close()

	// TODO(gri) importer.ImportData takes a []byte instead of an io.Reader;
	// hence the need to read some amount of data. At the same time we don't
	// want to read the entire, potentially very large object file. For now,
	// read 10K. Fix this!
	var data = make([]byte, 10<<10)
	n, err := reader.Read(data)
	if err != nil && err != io.EOF {
		return nil, err
	}

	return importer.ImportData(packages, data[:n])
}

// openGccgoExportFile was copied from gccgoimporter.
func openGccgoExportFile(fpath string) (reader io.ReadSeeker, closer io.Closer, err error) {
	f, err := os.Open(fpath)
	if err != nil {
		return
	}
	defer func() {
		if err != nil {
			f.Close()
		}
	}()
	closer = f

	var magic [4]byte
	_, err = f.ReadAt(magic[:], 0)
	if err != nil {
		return
	}

	if string(magic[:]) == "v1;\n" {
		// Raw export data.
		reader = f
		return
	}

	ef, err := elf.NewFile(f)
	if err != nil {
		return
	}

	sec := ef.Section(".go_export")
	if sec == nil {
		err = fmt.Errorf("%s: .go_export section not found", fpath)
		return
	}

	reader = sec.Open()
	return
}
