// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"fmt"
	"go/constant"
	"os"
	"path"
	"runtime"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/archive"
	"cmd/internal/bio"
	"cmd/internal/goobj"
	"cmd/internal/objabi"
	"cmd/internal/src"
)

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

func findpkg(name string) (file string, ok bool) {
	if islocalname(name) {
		if base.Flag.NoLocalImports {
			return "", false
		}

		if base.Flag.Cfg.PackageFile != nil {
			file, ok = base.Flag.Cfg.PackageFile[name]
			return file, ok
		}

		// try .a before .6.  important for building libraries:
		// if there is an array.6 in the array.a library,
		// want to find all of array.a, not just array.6.
		file = fmt.Sprintf("%s.a", name)
		if _, err := os.Stat(file); err == nil {
			return file, true
		}
		file = fmt.Sprintf("%s.o", name)
		if _, err := os.Stat(file); err == nil {
			return file, true
		}
		return "", false
	}

	// local imports should be canonicalized already.
	// don't want to see "encoding/../encoding/base64"
	// as different from "encoding/base64".
	if q := path.Clean(name); q != name {
		base.Errorf("non-canonical import path %q (should be %q)", name, q)
		return "", false
	}

	if base.Flag.Cfg.PackageFile != nil {
		file, ok = base.Flag.Cfg.PackageFile[name]
		return file, ok
	}

	for _, dir := range base.Flag.Cfg.ImportDirs {
		file = fmt.Sprintf("%s/%s.a", dir, name)
		if _, err := os.Stat(file); err == nil {
			return file, true
		}
		file = fmt.Sprintf("%s/%s.o", dir, name)
		if _, err := os.Stat(file); err == nil {
			return file, true
		}
	}

	if objabi.GOROOT != "" {
		suffix := ""
		suffixsep := ""
		if base.Flag.InstallSuffix != "" {
			suffixsep = "_"
			suffix = base.Flag.InstallSuffix
		} else if base.Flag.Race {
			suffixsep = "_"
			suffix = "race"
		} else if base.Flag.MSan {
			suffixsep = "_"
			suffix = "msan"
		}

		file = fmt.Sprintf("%s/pkg/%s_%s%s%s/%s.a", objabi.GOROOT, objabi.GOOS, objabi.GOARCH, suffixsep, suffix, name)
		if _, err := os.Stat(file); err == nil {
			return file, true
		}
		file = fmt.Sprintf("%s/pkg/%s_%s%s%s/%s.o", objabi.GOROOT, objabi.GOOS, objabi.GOARCH, suffixsep, suffix, name)
		if _, err := os.Stat(file); err == nil {
			return file, true
		}
	}

	return "", false
}

// myheight tracks the local package's height based on packages
// imported so far.
var myheight int

func importfile(f constant.Value) *types.Pkg {
	if f.Kind() != constant.String {
		base.Errorf("import path must be a string")
		return nil
	}

	path_ := constant.StringVal(f)
	if len(path_) == 0 {
		base.Errorf("import path is empty")
		return nil
	}

	if isbadimport(path_, false) {
		return nil
	}

	// The package name main is no longer reserved,
	// but we reserve the import path "main" to identify
	// the main package, just as we reserve the import
	// path "math" to identify the standard math package.
	if path_ == "main" {
		base.Errorf("cannot import \"main\"")
		base.ErrorExit()
	}

	if base.Ctxt.Pkgpath != "" && path_ == base.Ctxt.Pkgpath {
		base.Errorf("import %q while compiling that package (import cycle)", path_)
		base.ErrorExit()
	}

	if mapped, ok := base.Flag.Cfg.ImportMap[path_]; ok {
		path_ = mapped
	}

	if path_ == "unsafe" {
		return ir.Pkgs.Unsafe
	}

	if islocalname(path_) {
		if path_[0] == '/' {
			base.Errorf("import path cannot be absolute path")
			return nil
		}

		prefix := base.Ctxt.Pathname
		if base.Flag.D != "" {
			prefix = base.Flag.D
		}
		path_ = path.Join(prefix, path_)

		if isbadimport(path_, true) {
			return nil
		}
	}

	file, found := findpkg(path_)
	if !found {
		base.Errorf("can't find import: %q", path_)
		base.ErrorExit()
	}

	importpkg := types.NewPkg(path_, "")
	if importpkg.Imported {
		return importpkg
	}

	importpkg.Imported = true

	imp, err := bio.Open(file)
	if err != nil {
		base.Errorf("can't open import: %q: %v", path_, err)
		base.ErrorExit()
	}
	defer imp.Close()

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
	q := fmt.Sprintf("%s %s %s %s\n", objabi.GOOS, objabi.GOARCH, objabi.Version, objabi.Expstring())
	if p[10:] != q {
		base.Errorf("import %s: object is [%s] expected [%s]", file, p[10:], q)
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
		base.Errorf("cannot import %s: old export format no longer supported (recompile library)", path_)
		return nil

	case 'B':
		if base.Debug.Export != 0 {
			fmt.Printf("importing %s (%s)\n", path_, file)
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
		base.Errorf("no import in %q", path_)
		base.ErrorExit()
	}

	// assume files move (get installed) so don't record the full path
	if base.Flag.Cfg.PackageFile != nil {
		// If using a packageFile map, assume path_ can be recorded directly.
		base.Ctxt.AddImport(path_, fingerprint)
	} else {
		// For file "/Users/foo/go/pkg/darwin_amd64/math.a" record "math.a".
		base.Ctxt.AddImport(file[len(file)-len(path_)-len(".a"):], fingerprint)
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

func isbadimport(path string, allowSpace bool) bool {
	if strings.Contains(path, "\x00") {
		base.Errorf("import path contains NUL")
		return true
	}

	for _, ri := range reservedimports {
		if path == ri {
			base.Errorf("import path %q is reserved and cannot be used", path)
			return true
		}
	}

	for _, r := range path {
		if r == utf8.RuneError {
			base.Errorf("import path contains invalid UTF-8 sequence: %q", path)
			return true
		}

		if r < 0x20 || r == 0x7f {
			base.Errorf("import path contains control character: %q", path)
			return true
		}

		if r == '\\' {
			base.Errorf("import path contains backslash; use slash: %q", path)
			return true
		}

		if !allowSpace && unicode.IsSpace(r) {
			base.Errorf("import path contains space character: %q", path)
			return true
		}

		if strings.ContainsRune("!\"#$%&'()*,:;<=>?[]^`{|}", r) {
			base.Errorf("import path contains invalid character '%c': %q", r, path)
			return true
		}
	}

	return false
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
			// We'll report errors after type checking in checkDotImports.
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
