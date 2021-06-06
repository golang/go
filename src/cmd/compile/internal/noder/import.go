// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"errors"
	"fmt"
	"internal/buildcfg"
	"io"
	"os"
	pathpkg "path"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"

	"cmd/compile/internal/base"
	"cmd/compile/internal/importer"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/compile/internal/types2"
	"cmd/internal/archive"
	"cmd/internal/bio"
	"cmd/internal/goobj"
	"cmd/internal/objabi"
	"cmd/internal/src"
)

// Temporary import helper to get type2-based type-checking going.
type gcimports struct {
	packages map[string]*types2.Package
}

func (m *gcimports) Import(path string) (*types2.Package, error) {
	return m.ImportFrom(path, "" /* no vendoring */, 0)
}

func (m *gcimports) ImportFrom(path, srcDir string, mode types2.ImportMode) (*types2.Package, error) {
	if mode != 0 {
		panic("mode must be 0")
	}

	path, err := resolveImportPath(path)
	if err != nil {
		return nil, err
	}

	lookup := func(path string) (io.ReadCloser, error) { return openPackage(path) }
	return importer.Import(m.packages, path, srcDir, lookup)
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

// myheight tracks the local package's height based on packages
// imported so far.
var myheight int

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

	if base.Ctxt.Pkgpath != "" && path == base.Ctxt.Pkgpath {
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

// TODO(mdempsky): Return an error instead.
func importfile(decl *syntax.ImportDecl) *types.Pkg {
	if decl.Path.Kind != syntax.StringLit {
		base.Errorf("import path must be a string")
		return nil
	}

	path, err := strconv.Unquote(decl.Path.Value)
	if err != nil {
		base.Errorf("import path must be a string")
		return nil
	}

	if err := checkImportPath(path, false); err != nil {
		base.Errorf("%s", err.Error())
		return nil
	}

	path, err = resolveImportPath(path)
	if err != nil {
		base.Errorf("%s", err)
		return nil
	}

	importpkg := types.NewPkg(path, "")
	if importpkg.Direct {
		return importpkg // already fully loaded
	}
	importpkg.Direct = true
	typecheck.Target.Imports = append(typecheck.Target.Imports, importpkg)

	if path == "unsafe" {
		return importpkg // initialized with universe
	}

	f, err := openPackage(path)
	if err != nil {
		base.Errorf("could not import %q: %v", path, err)
		base.ErrorExit()
	}
	imp := bio.NewReader(f)
	defer imp.Close()
	file := f.Name()

	// check object header
	p, err := imp.ReadString('\n')
	if err != nil {
		base.Errorf("import %s: reading input: %v", file, err)
		base.ErrorExit()
	}

	if p == "!<arch>\n" { // package archive
		// package export block should be first
		sz := archive.ReadHeader(imp.Reader, "__.PKGDEF")
		if sz <= 0 {
			base.Errorf("import %s: not a package file", file)
			base.ErrorExit()
		}
		p, err = imp.ReadString('\n')
		if err != nil {
			base.Errorf("import %s: reading input: %v", file, err)
			base.ErrorExit()
		}
	}

	if !strings.HasPrefix(p, "go object ") {
		base.Errorf("import %s: not a go object file: %s", file, p)
		base.ErrorExit()
	}
	q := objabi.HeaderString()
	if p != q {
		base.Errorf("import %s: object is [%s] expected [%s]", file, p, q)
		base.ErrorExit()
	}

	// process header lines
	for {
		p, err = imp.ReadString('\n')
		if err != nil {
			base.Errorf("import %s: reading input: %v", file, err)
			base.ErrorExit()
		}
		if p == "\n" {
			break // header ends with blank line
		}
	}

	// Expect $$B\n to signal binary import format.

	// look for $$
	var c byte
	for {
		c, err = imp.ReadByte()
		if err != nil {
			break
		}
		if c == '$' {
			c, err = imp.ReadByte()
			if c == '$' || err != nil {
				break
			}
		}
	}

	// get character after $$
	if err == nil {
		c, _ = imp.ReadByte()
	}

	var fingerprint goobj.FingerprintType
	switch c {
	case '\n':
		base.Errorf("cannot import %s: old export format no longer supported (recompile library)", path)
		return nil

	case 'B':
		if base.Debug.Export != 0 {
			fmt.Printf("importing %s (%s)\n", path, file)
		}
		imp.ReadByte() // skip \n after $$B

		c, err = imp.ReadByte()
		if err != nil {
			base.Errorf("import %s: reading input: %v", file, err)
			base.ErrorExit()
		}

		// Indexed format is distinguished by an 'i' byte,
		// whereas previous export formats started with 'c', 'd', or 'v'.
		if c != 'i' {
			base.Errorf("import %s: unexpected package format byte: %v", file, c)
			base.ErrorExit()
		}
		fingerprint = typecheck.ReadImports(importpkg, imp)

	default:
		base.Errorf("no import in %q", path)
		base.ErrorExit()
	}

	// assume files move (get installed) so don't record the full path
	if base.Flag.Cfg.PackageFile != nil {
		// If using a packageFile map, assume path_ can be recorded directly.
		base.Ctxt.AddImport(path, fingerprint)
	} else {
		// For file "/Users/foo/go/pkg/darwin_amd64/math.a" record "math.a".
		base.Ctxt.AddImport(file[len(file)-len(path)-len(".a"):], fingerprint)
	}

	if importpkg.Height >= myheight {
		myheight = importpkg.Height + 1
	}

	return importpkg
}

// The linker uses the magic symbol prefixes "go." and "type."
// Avoid potential confusion between import paths and symbols
// by rejecting these reserved imports for now. Also, people
// "can do weird things in GOPATH and we'd prefer they didn't
// do _that_ weird thing" (per rsc). See also #4257.
var reservedimports = []string{
	"go",
	"type",
}

func checkImportPath(path string, allowSpace bool) error {
	if path == "" {
		return errors.New("import path is empty")
	}

	if strings.Contains(path, "\x00") {
		return errors.New("import path contains NUL")
	}

	for _, ri := range reservedimports {
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

func pkgnotused(lineno src.XPos, path string, name string) {
	// If the package was imported with a name other than the final
	// import path element, show it explicitly in the error message.
	// Note that this handles both renamed imports and imports of
	// packages containing unconventional package declarations.
	// Note that this uses / always, even on Windows, because Go import
	// paths always use forward slashes.
	elem := path
	if i := strings.LastIndex(elem, "/"); i >= 0 {
		elem = elem[i+1:]
	}
	if name == "" || elem == name {
		base.ErrorfAt(lineno, "imported and not used: %q", path)
	} else {
		base.ErrorfAt(lineno, "imported and not used: %q as %s", path, name)
	}
}

func mkpackage(pkgname string) {
	if types.LocalPkg.Name == "" {
		if pkgname == "_" {
			base.Errorf("invalid package name _")
		}
		types.LocalPkg.Name = pkgname
	} else {
		if pkgname != types.LocalPkg.Name {
			base.Errorf("package %s; expected %s", pkgname, types.LocalPkg.Name)
		}
	}
}

func clearImports() {
	type importedPkg struct {
		pos  src.XPos
		path string
		name string
	}
	var unused []importedPkg

	for _, s := range types.LocalPkg.Syms {
		n := ir.AsNode(s.Def)
		if n == nil {
			continue
		}
		if n.Op() == ir.OPACK {
			// throw away top-level package name left over
			// from previous file.
			// leave s->block set to cause redeclaration
			// errors if a conflicting top-level name is
			// introduced by a different file.
			p := n.(*ir.PkgName)
			if !p.Used && base.SyntaxErrors() == 0 {
				unused = append(unused, importedPkg{p.Pos(), p.Pkg.Path, s.Name})
			}
			s.Def = nil
			continue
		}
		if types.IsDotAlias(s) {
			// throw away top-level name left over
			// from previous import . "x"
			// We'll report errors after type checking in CheckDotImports.
			s.Def = nil
			continue
		}
	}

	sort.Slice(unused, func(i, j int) bool { return unused[i].pos.Before(unused[j].pos) })
	for _, pkg := range unused {
		pkgnotused(pkg.pos, pkg.path, pkg.name)
	}
}

// CheckDotImports reports errors for any unused dot imports.
func CheckDotImports() {
	for _, pack := range dotImports {
		if !pack.Used {
			base.ErrorfAt(pack.Pos(), "imported and not used: %q", pack.Pkg.Path)
		}
	}

	// No longer needed; release memory.
	dotImports = nil
	typecheck.DotImportRefs = nil
}

// dotImports tracks all PkgNames that have been dot-imported.
var dotImports []*ir.PkgName

// find all the exported symbols in package referenced by PkgName,
// and make them available in the current package
func importDot(pack *ir.PkgName) {
	if typecheck.DotImportRefs == nil {
		typecheck.DotImportRefs = make(map[*ir.Ident]*ir.PkgName)
	}

	opkg := pack.Pkg
	for _, s := range opkg.Syms {
		if s.Def == nil {
			if _, ok := typecheck.DeclImporter[s]; !ok {
				continue
			}
		}
		if !types.IsExported(s.Name) || strings.ContainsRune(s.Name, 0xb7) { // 0xb7 = center dot
			continue
		}
		s1 := typecheck.Lookup(s.Name)
		if s1.Def != nil {
			pkgerror := fmt.Sprintf("during import %q", opkg.Path)
			typecheck.Redeclared(base.Pos, s1, pkgerror)
			continue
		}

		id := ir.NewIdent(src.NoXPos, s)
		typecheck.DotImportRefs[id] = pack
		s1.Def = id
		s1.Block = 1
	}

	dotImports = append(dotImports, pack)
}

// importName is like oldname,
// but it reports an error if sym is from another package and not exported.
func importName(sym *types.Sym) ir.Node {
	n := oldname(sym)
	if !types.IsExported(sym.Name) && sym.Pkg != types.LocalPkg {
		n.SetDiag(true)
		base.Errorf("cannot refer to unexported name %s.%s", sym.Pkg.Name, sym.Name)
	}
	return n
}
