// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements access to gccgo-generated export data.

package main

import (
	"debug/elf"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"

	"code.google.com/p/go.tools/go/gccgoimporter"
	"code.google.com/p/go.tools/go/importer"
	"code.google.com/p/go.tools/go/types"
)

func init() {
	incpaths := []string{"/"}

	// importer for default gccgo
	var inst gccgoimporter.GccgoInstallation
	inst.InitFromDriver("gccgo")
	register("gccgo", inst.GetImporter(incpaths))

	// importer for gccgo using condensed export format (experimental)
	register("gccgo-new", getNewImporter(append(append(incpaths, inst.SearchPaths()...), ".")))
}

// This function is an adjusted variant of gccgoimporter.GccgoInstallation.GetImporter.
func getNewImporter(searchpaths []string) types.Importer {
	return func(imports map[string]*types.Package, pkgpath string) (pkg *types.Package, err error) {
		if pkgpath == "unsafe" {
			return types.Unsafe, nil
		}

		fpath, err := findExportFile(searchpaths, pkgpath)
		if err != nil {
			return
		}

		reader, closer, err := openExportFile(fpath)
		if err != nil {
			return nil, err
		}
		defer closer.Close()

		// TODO(gri) At the moment we just read the entire file.
		// We should change importer.ImportData to take an io.Reader instead.
		data, err := ioutil.ReadAll(reader)
		if err != nil && err != io.EOF {
			return nil, err
		}

		return importer.ImportData(packages, data)
	}
}

// This function is an exact copy of gccgoimporter.findExportFile.
func findExportFile(searchpaths []string, pkgpath string) (string, error) {
	for _, spath := range searchpaths {
		pkgfullpath := filepath.Join(spath, pkgpath)
		pkgdir, name := filepath.Split(pkgfullpath)

		for _, filepath := range [...]string{
			pkgfullpath,
			pkgfullpath + ".gox",
			pkgdir + "lib" + name + ".so",
			pkgdir + "lib" + name + ".a",
			pkgfullpath + ".o",
		} {
			fi, err := os.Stat(filepath)
			if err == nil && !fi.IsDir() {
				return filepath, nil
			}
		}
	}

	return "", fmt.Errorf("%s: could not find export data (tried %s)", pkgpath, strings.Join(searchpaths, ":"))
}

// This function is an exact copy of gccgoimporter.openExportFile.
func openExportFile(fpath string) (reader io.ReadSeeker, closer io.Closer, err error) {
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
