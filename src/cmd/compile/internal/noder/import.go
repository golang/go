// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"errors"
	"fmt"
	"internal/buildcfg"
	"internal/pkgbits"
	"os"
	pathpkg "path"
	"runtime"
	"strings"
	"unicode"
	"unicode/utf8"

	"cmd/compile/internal/base"
	"cmd/compile/internal/importer"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/compile/internal/types2"
	"cmd/internal/archive"
	"cmd/internal/bio"
	"cmd/internal/goobj"
	"cmd/internal/objabi"
)

type gcimports struct {
	ctxt     *types2.Context
	packages map[string]*types2.Package
}

func (m *gcimports) Import(path string) (*types2.Package, error) {
	return m.ImportFrom(path, "" /* no vendoring */, 0)
}

func (m *gcimports) ImportFrom(path, srcDir string, mode types2.ImportMode) (*types2.Package, error) {
	if mode != 0 {
		panic("mode must be 0")
	}

	_, pkg, err := readImportFile(path, typecheck.Target, m.ctxt, m.packages)
	return pkg, err
}

func isDriveLetter(b byte) bool {
	return 'a' <= b && b <= 'z' || 'A' <= b && b <= 'Z'
}

// is this path a local name? begins with ./ or ../ or /
func islocalname(name string) bool {
	return strings.HasPrefix(name, "/") ||
		runtime.GOOS == "windows" && len(name) >= 3 && isDriveLetter(name[0]) && name[1] == ':' && name[2] == '/' ||
		strings.HasPrefix(name, "./") || name == "." ||
		strings.HasPrefix(name, "../") || name == ".."
}

func openPackage(path string) (*os.File, error) {
	if islocalname(path) {
		if base.Flag.NoLocalImports {
			return nil, errors.New("local imports disallowed")
		}

		if base.Flag.Cfg.PackageFile != nil {
			return os.Open(base.Flag.Cfg.PackageFile[path])
		}

		// try .a before .o.  important for building libraries:
		// if there is an array.o in the array.a library,
		// want to find all of array.a, not just array.o.
		if file, err := os.Open(fmt.Sprintf("%s.a", path)); err == nil {
			return file, nil
		}
		if file, err := os.Open(fmt.Sprintf("%s.o", path)); err == nil {
			return file, nil
		}
		return nil, errors.New("file not found")
	}

	// local imports should be canonicalized already.
	// don't want to see "encoding/../encoding/base64"
	// as different from "encoding/base64".
	if q := pathpkg.Clean(path); q != path {
		return nil, fmt.Errorf("non-canonical import path %q (should be %q)", path, q)
	}

	if base.Flag.Cfg.PackageFile != nil {
		return os.Open(base.Flag.Cfg.PackageFile[path])
	}

	for _, dir := range base.Flag.Cfg.ImportDirs {
		if file, err := os.Open(fmt.Sprintf("%s/%s.a", dir, path)); err == nil {
			return file, nil
		}
		if file, err := os.Open(fmt.Sprintf("%s/%s.o", dir, path)); err == nil {
			return file, nil
		}
	}

	if buildcfg.GOROOT != "" {
		suffix := ""
		if base.Flag.InstallSuffix != "" {
			suffix = "_" + base.Flag.InstallSuffix
		} else if base.Flag.Race {
			suffix = "_race"
		} else if base.Flag.MSan {
			suffix = "_msan"
		} else if base.Flag.ASan {
			suffix = "_asan"
		}

		if file, err := os.Open(fmt.Sprintf("%s/pkg/%s_%s%s/%s.a", buildcfg.GOROOT, buildcfg.GOOS, buildcfg.GOARCH, suffix, path)); err == nil {
			return file, nil
		}
		if file, err := os.Open(fmt.Sprintf("%s/pkg/%s_%s%s/%s.o", buildcfg.GOROOT, buildcfg.GOOS, buildcfg.GOARCH, suffix, path)); err == nil {
			return file, nil
		}
	}
	return nil, errors.New("file not found")
}

// resolveImportPath resolves an import path as it appears in a Go
// source file to the package's full path.
func resolveImportPath(path string) (string, error) {
	// The package name main is no longer reserved,
	// but we reserve the import path "main" to identify
	// the main package, just as we reserve the import
	// path "math" to identify the standard math package.
	if path == "main" {
		return "", errors.New("cannot import \"main\"")
	}

	if base.Ctxt.Pkgpath == "" {
		panic("missing pkgpath")
	}
	if path == base.Ctxt.Pkgpath {
		return "", fmt.Errorf("import %q while compiling that package (import cycle)", path)
	}

	if mapped, ok := base.Flag.Cfg.ImportMap[path]; ok {
		path = mapped
	}

	if islocalname(path) {
		if path[0] == '/' {
			return "", errors.New("import path cannot be absolute path")
		}

		prefix := base.Flag.D
		if prefix == "" {
			// Questionable, but when -D isn't specified, historically we
			// resolve local import paths relative to the directory the
			// compiler's current directory, not the respective source
			// file's directory.
			prefix = base.Ctxt.Pathname
		}
		path = pathpkg.Join(prefix, path)

		if err := checkImportPath(path, true); err != nil {
			return "", err
		}
	}

	return path, nil
}

// readImportFile reads the import file for the given package path and
// returns its types.Pkg representation. If packages is non-nil, the
// types2.Package representation is also returned.
func readImportFile(path string, target *ir.Package, env *types2.Context, packages map[string]*types2.Package) (pkg1 *types.Pkg, pkg2 *types2.Package, err error) {
	path, err = resolveImportPath(path)
	if err != nil {
		return
	}

	if path == "unsafe" {
		pkg1, pkg2 = types.UnsafePkg, types2.Unsafe

		// TODO(mdempsky): Investigate if this actually matters. Why would
		// the linker or runtime care whether a package imported unsafe?
		if !pkg1.Direct {
			pkg1.Direct = true
			target.Imports = append(target.Imports, pkg1)
		}

		return
	}

	pkg1 = types.NewPkg(path, "")
	if packages != nil {
		pkg2 = packages[path]
		assert(pkg1.Direct == (pkg2 != nil && pkg2.Complete()))
	}

	if pkg1.Direct {
		return
	}
	pkg1.Direct = true
	target.Imports = append(target.Imports, pkg1)

	f, err := openPackage(path)
	if err != nil {
		return
	}
	defer f.Close()

	r, end, err := findExportData(f)
	if err != nil {
		return
	}

	if base.Debug.Export != 0 {
		fmt.Printf("importing %s (%s)\n", path, f.Name())
	}

	c, err := r.ReadByte()
	if err != nil {
		return
	}

	pos := r.Offset()

	// Map export data section into memory as a single large
	// string. This reduces heap fragmentation and allows returning
	// individual substrings very efficiently.
	var data string
	data, err = base.MapFile(r.File(), pos, end-pos)
	if err != nil {
		return
	}

	switch c {
	case 'u':
		// TODO(mdempsky): This seems a bit clunky.
		data = strings.TrimSuffix(data, "\n$$\n")

		pr := pkgbits.NewPkgDecoder(pkg1.Path, data)

		// Read package descriptors for both types2 and compiler backend.
		readPackage(newPkgReader(pr), pkg1, false)
		pkg2 = importer.ReadPackage(env, packages, pr)

	default:
		// Indexed format is distinguished by an 'i' byte,
		// whereas previous export formats started with 'c', 'd', or 'v'.
		err = fmt.Errorf("unexpected package format byte: %v", c)
		return
	}

	err = addFingerprint(path, f, end)
	return
}

// findExportData returns a *bio.Reader positioned at the start of the
// binary export data section, and a file offset for where to stop
// reading.
func findExportData(f *os.File) (r *bio.Reader, end int64, err error) {
	r = bio.NewReader(f)

	// check object header
	line, err := r.ReadString('\n')
	if err != nil {
		return
	}

	if line == "!<arch>\n" { // package archive
		// package export block should be first
		sz := int64(archive.ReadHeader(r.Reader, "__.PKGDEF"))
		if sz <= 0 {
			err = errors.New("not a package file")
			return
		}
		end = r.Offset() + sz
		line, err = r.ReadString('\n')
		if err != nil {
			return
		}
	} else {
		// Not an archive; provide end of file instead.
		// TODO(mdempsky): I don't think this happens anymore.
		var fi os.FileInfo
		fi, err = f.Stat()
		if err != nil {
			return
		}
		end = fi.Size()
	}

	if !strings.HasPrefix(line, "go object ") {
		err = fmt.Errorf("not a go object file: %s", line)
		return
	}
	if expect := objabi.HeaderString(); line != expect {
		err = fmt.Errorf("object is [%s] expected [%s]", line, expect)
		return
	}

	// process header lines
	for !strings.HasPrefix(line, "$$") {
		line, err = r.ReadString('\n')
		if err != nil {
			return
		}
	}

	// Expect $$B\n to signal binary import format.
	if line != "$$B\n" {
		err = errors.New("old export format no longer supported (recompile library)")
		return
	}

	return
}

// addFingerprint reads the linker fingerprint included at the end of
// the exportdata.
func addFingerprint(path string, f *os.File, end int64) error {
	const eom = "\n$$\n"
	var fingerprint goobj.FingerprintType

	var buf [len(fingerprint) + len(eom)]byte
	if _, err := f.ReadAt(buf[:], end-int64(len(buf))); err != nil {
		return err
	}

	// Caller should have given us the end position of the export data,
	// which should end with the "\n$$\n" marker. As a consistency check
	// to make sure we're reading at the right offset, make sure we
	// found the marker.
	if s := string(buf[len(fingerprint):]); s != eom {
		return fmt.Errorf("expected $$ marker, but found %q", s)
	}

	copy(fingerprint[:], buf[:])
	base.Ctxt.AddImport(path, fingerprint)

	return nil
}

func checkImportPath(path string, allowSpace bool) error {
	if path == "" {
		return errors.New("import path is empty")
	}

	if strings.Contains(path, "\x00") {
		return errors.New("import path contains NUL")
	}

	for ri := range base.ReservedImports {
		if path == ri {
			return fmt.Errorf("import path %q is reserved and cannot be used", path)
		}
	}

	for _, r := range path {
		switch {
		case r == utf8.RuneError:
			return fmt.Errorf("import path contains invalid UTF-8 sequence: %q", path)
		case r < 0x20 || r == 0x7f:
			return fmt.Errorf("import path contains control character: %q", path)
		case r == '\\':
			return fmt.Errorf("import path contains backslash; use slash: %q", path)
		case !allowSpace && unicode.IsSpace(r):
			return fmt.Errorf("import path contains space character: %q", path)
		case strings.ContainsRune("!\"#$%&'()*,:;<=>?[]^`{|}", r):
			return fmt.Errorf("import path contains invalid character '%c': %q", r, path)
		}
	}

	return nil
}
