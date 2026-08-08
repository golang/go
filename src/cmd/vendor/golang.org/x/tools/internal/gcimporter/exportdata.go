// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file should be kept in sync with $GOROOT/src/internal/exportdata/exportdata.go.
// This file also additionally implements FindExportData for gcexportdata.NewReader.

package gcimporter

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"go/build"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
)

// FindExportData positions the reader r at the beginning of the
// export data section of an underlying cmd/compile created archive
// file by reading from it. The reader must be positioned at the
// start of the file before calling this function.
// This returns the length of the export data in bytes.
//
// This function is needed by [gcexportdata.Read], which must
// accept inputs produced by the last two releases of cmd/compile,
// plus tip.
func FindExportData(r *bufio.Reader) (size int64, err error) {
	arsize, err := FindPackageDefinition(r)
	if err != nil {
		return
	}
	size = int64(arsize)

	objapi, headers, err := ReadObjectHeaders(r)
	if err != nil {
		return
	}
	size -= int64(len(objapi))
	for _, h := range headers {
		size -= int64(len(h))
	}

	// Check for the binary export data section header "$$B\n".
	// TODO(taking): Unify with ReadExportDataHeader so that it stops at the 'u' instead of reading
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
	// ugly so gcimporter gives up enforcing this. The compiler and go/types
	// importer do enforce this, which seems good enough.
	const endofsection = "\n$$\n"
	size -= int64(len(endofsection))

	if size < 0 {
		err = fmt.Errorf("invalid size (%d) in the archive file: %d bytes remain without section headers (recompile package)", arsize, size)
		return
	}

	return
}

// ReadUnified reads the contents of the unified export data from a reader r
// that contains the contents of a GC-created archive file.
//
// On success, the reader will be positioned after the end-of-section marker "\n$$\n".
//
// Supported GC-created archive files have 4 layers of nesting:
//   - An archive file containing a package definition file.
//   - The package definition file contains headers followed by a data section.
//     Headers are lines (≤ 4kb) that do not start with "$$".
//   - The data section starts with "$$B\n" followed by export data followed
//     by an end of section marker "\n$$\n". (The section start "$$\n" is no
//     longer supported.)
//   - The export data starts with a format byte ('u') followed by the <data> in
//     the given format. (See ReadExportDataHeader for older formats.)
//
// Putting this together, the bytes in a GC-created archive files are expected
// to look like the following.
// See cmd/internal/archive for more details on ar file headers.
//
// | <!arch>\n             | ar file signature
// | __.PKGDEF...size...\n | ar header for __.PKGDEF including size.
// | go object <...>\n     | objabi header
// | <optional headers>\n  | other headers such as build id
// | $$B\n                 | binary format marker
// | u<data>\n             | unified export <data>
// | $$\n                  | end-of-section marker
// | [optional padding]    | padding byte (0x0A) if size is odd
// | [ar file header]      | other ar files
// | [ar file data]        |
func ReadUnified(r *bufio.Reader) (data []byte, err error) {
	// We historically guaranteed headers at the default buffer size (4096) work.
	// This ensures we can use ReadSlice throughout.
	const minBufferSize = 4096
	r = bufio.NewReaderSize(r, minBufferSize)

	size, err := FindPackageDefinition(r)
	if err != nil {
		return
	}
	n := size

	objapi, headers, err := ReadObjectHeaders(r)
	if err != nil {
		return
	}
	n -= len(objapi)
	for _, h := range headers {
		n -= len(h)
	}

	hdrlen, err := ReadExportDataHeader(r)
	if err != nil {
		return
	}
	n -= hdrlen

	// size also includes the end of section marker. Remove that many bytes from the end.
	const marker = "\n$$\n"
	n -= len(marker)

	if n < 0 {
		err = fmt.Errorf("invalid size (%d) in the archive file: %d bytes remain without section headers (recompile package)", size, n)
		return
	}

	// Read n bytes from buf.
	data = make([]byte, n)
	_, err = io.ReadFull(r, data)
	if err != nil {
		return
	}

	// Check for marker at the end.
	var suffix [len(marker)]byte
	_, err = io.ReadFull(r, suffix[:])
	if err != nil {
		return
	}
	if s := string(suffix[:]); s != marker {
		err = fmt.Errorf("read %q instead of end-of-section marker (%q)", s, marker)
		return
	}

	return
}

// FindPackageDefinition positions the reader r at the beginning of a package
// definition file ("__.PKGDEF") within a GC-created archive by reading
// from it, and returns the size of the package definition file in the archive.
//
// The reader must be positioned at the start of the archive file before calling
// this function, and "__.PKGDEF" is assumed to be the first file in the archive.
//
// See cmd/internal/archive for details on the archive format.
func FindPackageDefinition(r *bufio.Reader) (size int, err error) {
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

// ReadObjectHeaders reads object headers from the reader. Object headers are
// lines that do not start with an end-of-section marker "$$". The first header
// is the objabi header. On success, the reader will be positioned at the beginning
// of the end-of-section marker.
//
// It returns an error if any header does not fit in r.Size() bytes.
func ReadObjectHeaders(r *bufio.Reader) (objapi string, headers []string, err error) {
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

// ReadExportDataHeader reads the export data header and format from r.
// It returns the number of bytes read, or an error if the format is no longer
// supported or it failed to read.
//
// The only currently supported format is binary export data in the
// unified export format.
func ReadExportDataHeader(r *bufio.Reader) (n int, err error) {
	// Read export data header.
	line, err := r.ReadSlice('\n')
	if err != nil {
		return
	}

	hdr := string(line)
	switch hdr {
	case "$$\n":
		err = fmt.Errorf("old textual export format no longer supported (recompile package)")
		return

	case "$$B\n":
		var format byte
		format, err = r.ReadByte()
		if err != nil {
			return
		}
		// The unified export format starts with a 'u'.
		switch format {
		case 'u':
		default:
			// Older no longer supported export formats include:
			// indexed export format which started with an 'i'; and
			// the older binary export format which started with a 'c',
			// 'd', or 'v' (from "version").
			err = fmt.Errorf("binary export format %q is no longer supported (recompile package)", format)
			return
		}

	default:
		err = fmt.Errorf("unknown export data header: %q", hdr)
		return
	}

	n = len(hdr) + 1 // + 1 is for 'u'
	return
}

// FindPkg returns the filename and unique package id for an import
// path based on package information provided by build.Import (using
// the build.Default build.Context). A relative srcDir is interpreted
// relative to the current working directory.
//
// FindPkg is only used in tests within x/tools.
func FindPkg(path, srcDir string) (filename, id string, err error) {
	// TODO(taking): Move internal/exportdata.FindPkg into its own file,
	// and then this copy into a _test package.
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

var pkgExts = [...]string{".a", ".o"} // a file from the build cache will have no extension

var exportMap sync.Map // package dir → func() (string, error)

// lookupGorootExport returns the location of the export data
// (normally found in the build cache, but located in GOROOT/pkg
// in prior Go releases) for the package located in pkgDir.
//
// (We use the package's directory instead of its import path
// mainly to simplify handling of the packages in src/vendor
// and cmd/vendor.)
//
// lookupGorootExport is only used in tests within x/tools.
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
