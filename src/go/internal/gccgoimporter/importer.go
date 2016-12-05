// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package gccgoimporter implements Import for gccgo-generated object files.
package gccgoimporter // import "go/internal/gccgoimporter"

import (
	"bytes"
	"debug/elf"
	"fmt"
	"go/types"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// A PackageInit describes an imported package that needs initialization.
type PackageInit struct {
	Name     string // short package name
	InitFunc string // name of init function
	Priority int    // priority of init function, see InitData.Priority
}

// The gccgo-specific init data for a package.
type InitData struct {
	// Initialization priority of this package relative to other packages.
	// This is based on the maximum depth of the package's dependency graph;
	// it is guaranteed to be greater than that of its dependencies.
	Priority int

	// The list of packages which this package depends on to be initialized,
	// including itself if needed. This is the subset of the transitive closure of
	// the package's dependencies that need initialization.
	Inits []PackageInit
}

// Locate the file from which to read export data.
// This is intended to replicate the logic in gofrontend.
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

const (
	gccgov1Magic    = "v1;\n"
	gccgov2Magic    = "v2;\n"
	goimporterMagic = "\n$$ "
	archiveMagic    = "!<ar"
)

// Opens the export data file at the given path. If this is an ELF file,
// searches for and opens the .go_export section. If this is an archive,
// reads the export data from the first member, which is assumed to be an ELF file.
// This is intended to replicate the logic in gofrontend.
func openExportFile(fpath string) (reader io.ReadSeeker, closer io.Closer, err error) {
	f, err := os.Open(fpath)
	if err != nil {
		return
	}
	closer = f
	defer func() {
		if err != nil && closer != nil {
			f.Close()
		}
	}()

	var magic [4]byte
	_, err = f.ReadAt(magic[:], 0)
	if err != nil {
		return
	}

	var elfreader io.ReaderAt
	switch string(magic[:]) {
	case gccgov1Magic, gccgov2Magic, goimporterMagic:
		// Raw export data.
		reader = f
		return

	case archiveMagic:
		// TODO(pcc): Read the archive directly instead of using "ar".
		f.Close()
		closer = nil

		cmd := exec.Command("ar", "p", fpath)
		var out []byte
		out, err = cmd.Output()
		if err != nil {
			return
		}

		elfreader = bytes.NewReader(out)

	default:
		elfreader = f
	}

	ef, err := elf.NewFile(elfreader)
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

// An Importer resolves import paths to Packages. The imports map records
// packages already known, indexed by package path.
// An importer must determine the canonical package path and check imports
// to see if it is already present in the map. If so, the Importer can return
// the map entry. Otherwise, the importer must load the package data for the
// given path into a new *Package, record it in imports map, and return the
// package.
type Importer func(imports map[string]*types.Package, path string) (*types.Package, error)

func GetImporter(searchpaths []string, initmap map[*types.Package]InitData) Importer {
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
			return
		}
		if closer != nil {
			defer closer.Close()
		}

		var magic [4]byte
		_, err = reader.Read(magic[:])
		if err != nil {
			return
		}
		_, err = reader.Seek(0, io.SeekStart)
		if err != nil {
			return
		}

		switch string(magic[:]) {
		case gccgov1Magic, gccgov2Magic:
			var p parser
			p.init(fpath, reader, imports)
			pkg = p.parsePackage()
			if initmap != nil {
				initmap[pkg] = p.initdata
			}

		// Excluded for now: Standard gccgo doesn't support this import format currently.
		// case goimporterMagic:
		// 	var data []byte
		// 	data, err = ioutil.ReadAll(reader)
		// 	if err != nil {
		// 		return
		// 	}
		// 	var n int
		// 	n, pkg, err = importer.ImportData(imports, data)
		// 	if err != nil {
		// 		return
		// 	}

		// 	if initmap != nil {
		// 		suffixreader := bytes.NewReader(data[n:])
		// 		var p parser
		// 		p.init(fpath, suffixreader, nil)
		// 		p.parseInitData()
		// 		initmap[pkg] = p.initdata
		// 	}

		default:
			err = fmt.Errorf("unrecognized magic string: %q", string(magic[:]))
		}

		return
	}
}
