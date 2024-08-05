// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// package importer implements Import for gc-generated object files.
package importer

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"go/build"
	"internal/pkgbits"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"

	"cmd/compile/internal/types2"
)

var exportMap sync.Map // package dir → func() (string, error)

// lookupGorootExport returns the location of the export data
// (normally found in the build cache, but located in GOROOT/pkg
// in prior Go releases) for the package located in pkgDir.
//
// (We use the package's directory instead of its import path
// mainly to simplify handling of the packages in src/vendor
// and cmd/vendor.)
func lookupGorootExport(pkgDir string) (string, error) {
	f, ok := exportMap.Load(pkgDir)
	if !ok {
		var (
			listOnce   sync.Once
			exportPath string
			err        error
		)
		f, _ = exportMap.LoadOrStore(pkgDir, func() (string, error) {
			listOnce.Do(func() {
				cmd := exec.Command(filepath.Join(build.Default.GOROOT, "bin", "go"), "list", "-export", "-f", "{{.Export}}", pkgDir)
				cmd.Dir = build.Default.GOROOT
				cmd.Env = append(os.Environ(), "PWD="+cmd.Dir, "GOROOT="+build.Default.GOROOT)
				var output []byte
				output, err = cmd.Output()
				if err != nil {
					if ee, ok := err.(*exec.ExitError); ok && len(ee.Stderr) > 0 {
						err = errors.New(string(ee.Stderr))
					}
					return
				}

				exports := strings.Split(string(bytes.TrimSpace(output)), "\n")
				if len(exports) != 1 {
					err = fmt.Errorf("go list reported %d exports; expected 1", len(exports))
					return
				}

				exportPath = exports[0]
			})

			return exportPath, err
		})
	}

	return f.(func() (string, error))()
}

var pkgExts = [...]string{".a", ".o"} // a file from the build cache will have no extension

// FindPkg returns the filename and unique package id for an import
// path based on package information provided by build.Import (using
// the build.Default build.Context). A relative srcDir is interpreted
// relative to the current working directory.
func FindPkg(path, srcDir string) (filename, id string, err error) {
	if path == "" {
		return "", "", errors.New("path is empty")
	}

	var noext string
	switch {
	default:
		// "x" -> "$GOPATH/pkg/$GOOS_$GOARCH/x.ext", "x"
		// Don't require the source files to be present.
		if abs, err := filepath.Abs(srcDir); err == nil { // see issue 14282
			srcDir = abs
		}
		var bp *build.Package
		bp, err = build.Import(path, srcDir, build.FindOnly|build.AllowBinary)
		if bp.PkgObj == "" {
			if bp.Goroot && bp.Dir != "" {
				filename, err = lookupGorootExport(bp.Dir)
				if err == nil {
					_, err = os.Stat(filename)
				}
				if err == nil {
					return filename, bp.ImportPath, nil
				}
			}
			goto notfound
		} else {
			noext = strings.TrimSuffix(bp.PkgObj, ".a")
		}
		id = bp.ImportPath

	case build.IsLocalImport(path):
		// "./x" -> "/this/directory/x.ext", "/this/directory/x"
		noext = filepath.Join(srcDir, path)
		id = noext

	case filepath.IsAbs(path):
		// for completeness only - go/build.Import
		// does not support absolute imports
		// "/x" -> "/x.ext", "/x"
		noext = path
		id = path
	}

	if false { // for debugging
		if path != id {
			fmt.Printf("%s -> %s\n", path, id)
		}
	}

	// try extensions
	for _, ext := range pkgExts {
		filename = noext + ext
		f, statErr := os.Stat(filename)
		if statErr == nil && !f.IsDir() {
			return filename, id, nil
		}
		if err == nil {
			err = statErr
		}
	}

notfound:
	if err == nil {
		return "", path, fmt.Errorf("can't find import: %q", path)
	}
	return "", path, fmt.Errorf("can't find import: %q: %w", path, err)
}

// Import imports a gc-generated package given its import path and srcDir, adds
// the corresponding package object to the packages map, and returns the object.
// The packages map must contain all packages already imported.
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
		filename, id, err = FindPkg(path, srcDir)
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
	hdr, size, err := FindExportData(buf)
	if err != nil {
		return
	}

	switch hdr {
	case "$$\n":
		err = fmt.Errorf("import %q: old textual export format no longer supported (recompile library)", path)

	case "$$B\n":
		var data []byte
		var r io.Reader = buf
		if size >= 0 {
			r = io.LimitReader(r, int64(size))
		}
		data, err = io.ReadAll(r)
		if err != nil {
			break
		}

		if len(data) == 0 {
			err = fmt.Errorf("import %q: missing export data", path)
			break
		}
		exportFormat := data[0]
		s := string(data[1:])

		// The indexed export format starts with an 'i'; the older
		// binary export format starts with a 'c', 'd', or 'v'
		// (from "version"). Select appropriate importer.
		switch exportFormat {
		case 'u':
			s = s[:strings.Index(s, "\n$$\n")]
			input := pkgbits.NewPkgDecoder(id, s)
			pkg = ReadPackage(nil, packages, input)
		case 'i':
			pkg, err = ImportData(packages, s, id)
		default:
			err = fmt.Errorf("import %q: old binary export format no longer supported (recompile library)", path)
		}

	default:
		err = fmt.Errorf("import %q: unknown export data header: %q", path, hdr)
	}

	return
}
