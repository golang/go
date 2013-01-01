// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package build

import (
	"bytes"
	"errors"
	"fmt"
	"go/ast"
	"go/doc"
	"go/parser"
	"go/token"
	"io"
	"io/ioutil"
	"log"
	"os"
	pathpkg "path"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"unicode"
)

// A Context specifies the supporting context for a build.
type Context struct {
	GOARCH      string   // target architecture
	GOOS        string   // target operating system
	GOROOT      string   // Go root
	GOPATH      string   // Go path
	CgoEnabled  bool     // whether cgo can be used
	BuildTags   []string // additional tags to recognize in +build lines
	InstallTag  string   // package install directory suffix
	UseAllFiles bool     // use files regardless of +build lines, file names
	Compiler    string   // compiler to assume when computing target paths

	// By default, Import uses the operating system's file system calls
	// to read directories and files.  To read from other sources,
	// callers can set the following functions.  They all have default
	// behaviors that use the local file system, so clients need only set
	// the functions whose behaviors they wish to change.

	// JoinPath joins the sequence of path fragments into a single path.
	// If JoinPath is nil, Import uses filepath.Join.
	JoinPath func(elem ...string) string

	// SplitPathList splits the path list into a slice of individual paths.
	// If SplitPathList is nil, Import uses filepath.SplitList.
	SplitPathList func(list string) []string

	// IsAbsPath reports whether path is an absolute path.
	// If IsAbsPath is nil, Import uses filepath.IsAbs.
	IsAbsPath func(path string) bool

	// IsDir reports whether the path names a directory.
	// If IsDir is nil, Import calls os.Stat and uses the result's IsDir method.
	IsDir func(path string) bool

	// HasSubdir reports whether dir is a subdirectory of
	// (perhaps multiple levels below) root.
	// If so, HasSubdir sets rel to a slash-separated path that
	// can be joined to root to produce a path equivalent to dir.
	// If HasSubdir is nil, Import uses an implementation built on
	// filepath.EvalSymlinks.
	HasSubdir func(root, dir string) (rel string, ok bool)

	// ReadDir returns a slice of os.FileInfo, sorted by Name,
	// describing the content of the named directory.
	// If ReadDir is nil, Import uses ioutil.ReadDir.
	ReadDir func(dir string) (fi []os.FileInfo, err error)

	// OpenFile opens a file (not a directory) for reading.
	// If OpenFile is nil, Import uses os.Open.
	OpenFile func(path string) (r io.ReadCloser, err error)
}

// joinPath calls ctxt.JoinPath (if not nil) or else filepath.Join.
func (ctxt *Context) joinPath(elem ...string) string {
	if f := ctxt.JoinPath; f != nil {
		return f(elem...)
	}
	return filepath.Join(elem...)
}

// splitPathList calls ctxt.SplitPathList (if not nil) or else filepath.SplitList.
func (ctxt *Context) splitPathList(s string) []string {
	if f := ctxt.SplitPathList; f != nil {
		return f(s)
	}
	return filepath.SplitList(s)
}

// isAbsPath calls ctxt.IsAbsSPath (if not nil) or else filepath.IsAbs.
func (ctxt *Context) isAbsPath(path string) bool {
	if f := ctxt.IsAbsPath; f != nil {
		return f(path)
	}
	return filepath.IsAbs(path)
}

// isDir calls ctxt.IsDir (if not nil) or else uses os.Stat.
func (ctxt *Context) isDir(path string) bool {
	if f := ctxt.IsDir; f != nil {
		return f(path)
	}
	fi, err := os.Stat(path)
	return err == nil && fi.IsDir()
}

// hasSubdir calls ctxt.HasSubdir (if not nil) or else uses
// the local file system to answer the question.
func (ctxt *Context) hasSubdir(root, dir string) (rel string, ok bool) {
	if f := ctxt.HasSubdir; f != nil {
		return f(root, dir)
	}

	if p, err := filepath.EvalSymlinks(root); err == nil {
		root = p
	}
	if p, err := filepath.EvalSymlinks(dir); err == nil {
		dir = p
	}
	const sep = string(filepath.Separator)
	root = filepath.Clean(root)
	if !strings.HasSuffix(root, sep) {
		root += sep
	}
	dir = filepath.Clean(dir)
	if !strings.HasPrefix(dir, root) {
		return "", false
	}
	return filepath.ToSlash(dir[len(root):]), true
}

// readDir calls ctxt.ReadDir (if not nil) or else ioutil.ReadDir.
func (ctxt *Context) readDir(path string) ([]os.FileInfo, error) {
	if f := ctxt.ReadDir; f != nil {
		return f(path)
	}
	return ioutil.ReadDir(path)
}

// openFile calls ctxt.OpenFile (if not nil) or else os.Open.
func (ctxt *Context) openFile(path string) (io.ReadCloser, error) {
	if fn := ctxt.OpenFile; fn != nil {
		return fn(path)
	}

	f, err := os.Open(path)
	if err != nil {
		return nil, err // nil interface
	}
	return f, nil
}

// isFile determines whether path is a file by trying to open it.
// It reuses openFile instead of adding another function to the
// list in Context.
func (ctxt *Context) isFile(path string) bool {
	f, err := ctxt.openFile(path)
	if err != nil {
		return false
	}
	f.Close()
	return true
}

// gopath returns the list of Go path directories.
func (ctxt *Context) gopath() []string {
	var all []string
	for _, p := range ctxt.splitPathList(ctxt.GOPATH) {
		if p == "" || p == ctxt.GOROOT {
			// Empty paths are uninteresting.
			// If the path is the GOROOT, ignore it.
			// People sometimes set GOPATH=$GOROOT, which is useless
			// but would cause us to find packages with import paths
			// like "pkg/math".
			// Do not get confused by this common mistake.
			continue
		}
		all = append(all, p)
	}
	return all
}

// SrcDirs returns a list of package source root directories.
// It draws from the current Go root and Go path but omits directories
// that do not exist.
func (ctxt *Context) SrcDirs() []string {
	var all []string
	if ctxt.GOROOT != "" {
		dir := ctxt.joinPath(ctxt.GOROOT, "src", "pkg")
		if ctxt.isDir(dir) {
			all = append(all, dir)
		}
	}
	for _, p := range ctxt.gopath() {
		dir := ctxt.joinPath(p, "src")
		if ctxt.isDir(dir) {
			all = append(all, dir)
		}
	}
	return all
}

// Default is the default Context for builds.
// It uses the GOARCH, GOOS, GOROOT, and GOPATH environment variables
// if set, or else the compiled code's GOARCH, GOOS, and GOROOT.
var Default Context = defaultContext()

var cgoEnabled = map[string]bool{
	"darwin/386":    true,
	"darwin/amd64":  true,
	"freebsd/386":   true,
	"freebsd/amd64": true,
	"linux/386":     true,
	"linux/amd64":   true,
	"linux/arm":     true,
	"netbsd/386":    true,
	"netbsd/amd64":  true,
	"openbsd/386":   true,
	"openbsd/amd64": true,
	"windows/386":   true,
	"windows/amd64": true,
}

func defaultContext() Context {
	var c Context

	c.GOARCH = envOr("GOARCH", runtime.GOARCH)
	c.GOOS = envOr("GOOS", runtime.GOOS)
	c.GOROOT = runtime.GOROOT()
	c.GOPATH = envOr("GOPATH", "")
	c.Compiler = runtime.Compiler

	switch os.Getenv("CGO_ENABLED") {
	case "1":
		c.CgoEnabled = true
	case "0":
		c.CgoEnabled = false
	default:
		c.CgoEnabled = cgoEnabled[c.GOOS+"/"+c.GOARCH]
	}

	return c
}

func envOr(name, def string) string {
	s := os.Getenv(name)
	if s == "" {
		return def
	}
	return s
}

// An ImportMode controls the behavior of the Import method.
type ImportMode uint

const (
	// If FindOnly is set, Import stops after locating the directory
	// that should contain the sources for a package.  It does not
	// read any files in the directory.
	FindOnly ImportMode = 1 << iota

	// If AllowBinary is set, Import can be satisfied by a compiled
	// package object without corresponding sources.
	AllowBinary
)

// A Package describes the Go package found in a directory.
type Package struct {
	Dir        string // directory containing package sources
	Name       string // package name
	Doc        string // documentation synopsis
	ImportPath string // import path of package ("" if unknown)
	Root       string // root of Go tree where this package lives
	SrcRoot    string // package source root directory ("" if unknown)
	PkgRoot    string // package install root directory ("" if unknown)
	BinDir     string // command install directory ("" if unknown)
	Goroot     bool   // package found in Go root
	PkgObj     string // installed .a file

	// Source files
	GoFiles      []string // .go source files (excluding CgoFiles, TestGoFiles, XTestGoFiles)
	CgoFiles     []string // .go source files that import "C"
	CFiles       []string // .c source files
	HFiles       []string // .h source files
	SFiles       []string // .s source files
	SysoFiles    []string // .syso system object files to add to archive
	SwigFiles    []string // .swig files
	SwigCXXFiles []string // .swigcxx files

	// Cgo directives
	CgoPkgConfig []string // Cgo pkg-config directives
	CgoCFLAGS    []string // Cgo CFLAGS directives
	CgoLDFLAGS   []string // Cgo LDFLAGS directives

	// Dependency information
	Imports   []string                    // imports from GoFiles, CgoFiles
	ImportPos map[string][]token.Position // line information for Imports

	// Test information
	TestGoFiles    []string                    // _test.go files in package
	TestImports    []string                    // imports from TestGoFiles
	TestImportPos  map[string][]token.Position // line information for TestImports
	XTestGoFiles   []string                    // _test.go files outside package
	XTestImports   []string                    // imports from XTestGoFiles
	XTestImportPos map[string][]token.Position // line information for XTestImports
}

// IsCommand reports whether the package is considered a
// command to be installed (not just a library).
// Packages named "main" are treated as commands.
func (p *Package) IsCommand() bool {
	return p.Name == "main"
}

// ImportDir is like Import but processes the Go package found in
// the named directory.
func (ctxt *Context) ImportDir(dir string, mode ImportMode) (*Package, error) {
	return ctxt.Import(".", dir, mode)
}

// NoGoError is the error used by Import to describe a directory
// containing no Go source files.
type NoGoError struct {
	Dir string
}

func (e *NoGoError) Error() string {
	return "no Go source files in " + e.Dir
}

// Import returns details about the Go package named by the import path,
// interpreting local import paths relative to the srcDir directory.
// If the path is a local import path naming a package that can be imported
// using a standard import path, the returned package will set p.ImportPath
// to that path.
//
// In the directory containing the package, .go, .c, .h, and .s files are
// considered part of the package except for:
//
//	- .go files in package documentation
//	- files starting with _ or . (likely editor temporary files)
//	- files with build constraints not satisfied by the context
//
// If an error occurs, Import returns a non-nil error and a non-nil
// *Package containing partial information.
//
func (ctxt *Context) Import(path string, srcDir string, mode ImportMode) (*Package, error) {
	p := &Package{
		ImportPath: path,
	}
	if path == "" {
		return p, fmt.Errorf("import %q: invalid import path", path)
	}

	var pkga string
	var pkgerr error
	switch ctxt.Compiler {
	case "gccgo":
		dir, elem := pathpkg.Split(p.ImportPath)
		pkga = "pkg/gccgo/" + dir + "lib" + elem + ".a"
	case "gc":
		tag := ""
		if ctxt.InstallTag != "" {
			tag = "_" + ctxt.InstallTag
		}
		pkga = "pkg/" + ctxt.GOOS + "_" + ctxt.GOARCH + tag + "/" + p.ImportPath + ".a"
	default:
		// Save error for end of function.
		pkgerr = fmt.Errorf("import %q: unknown compiler %q", path, ctxt.Compiler)
	}

	binaryOnly := false
	if IsLocalImport(path) {
		pkga = "" // local imports have no installed path
		if srcDir == "" {
			return p, fmt.Errorf("import %q: import relative to unknown directory", path)
		}
		if !ctxt.isAbsPath(path) {
			p.Dir = ctxt.joinPath(srcDir, path)
		}
		// Determine canonical import path, if any.
		if ctxt.GOROOT != "" {
			root := ctxt.joinPath(ctxt.GOROOT, "src", "pkg")
			if sub, ok := ctxt.hasSubdir(root, p.Dir); ok {
				p.Goroot = true
				p.ImportPath = sub
				p.Root = ctxt.GOROOT
				goto Found
			}
		}
		all := ctxt.gopath()
		for i, root := range all {
			rootsrc := ctxt.joinPath(root, "src")
			if sub, ok := ctxt.hasSubdir(rootsrc, p.Dir); ok {
				// We found a potential import path for dir,
				// but check that using it wouldn't find something
				// else first.
				if ctxt.GOROOT != "" {
					if dir := ctxt.joinPath(ctxt.GOROOT, "src", "pkg", sub); ctxt.isDir(dir) {
						goto Found
					}
				}
				for _, earlyRoot := range all[:i] {
					if dir := ctxt.joinPath(earlyRoot, "src", sub); ctxt.isDir(dir) {
						goto Found
					}
				}

				// sub would not name some other directory instead of this one.
				// Record it.
				p.ImportPath = sub
				p.Root = root
				goto Found
			}
		}
		// It's okay that we didn't find a root containing dir.
		// Keep going with the information we have.
	} else {
		if strings.HasPrefix(path, "/") {
			return p, fmt.Errorf("import %q: cannot import absolute path", path)
		}

		// tried records the location of unsucsessful package lookups
		var tried struct {
			goroot string
			gopath []string
		}

		// Determine directory from import path.
		if ctxt.GOROOT != "" {
			dir := ctxt.joinPath(ctxt.GOROOT, "src", "pkg", path)
			isDir := ctxt.isDir(dir)
			binaryOnly = !isDir && mode&AllowBinary != 0 && pkga != "" && ctxt.isFile(ctxt.joinPath(ctxt.GOROOT, pkga))
			if isDir || binaryOnly {
				p.Dir = dir
				p.Goroot = true
				p.Root = ctxt.GOROOT
				goto Found
			}
			tried.goroot = dir
		}
		for _, root := range ctxt.gopath() {
			dir := ctxt.joinPath(root, "src", path)
			isDir := ctxt.isDir(dir)
			binaryOnly = !isDir && mode&AllowBinary != 0 && pkga != "" && ctxt.isFile(ctxt.joinPath(root, pkga))
			if isDir || binaryOnly {
				p.Dir = dir
				p.Root = root
				goto Found
			}
			tried.gopath = append(tried.gopath, dir)
		}

		// package was not found
		var paths []string
		if tried.goroot != "" {
			paths = append(paths, fmt.Sprintf("\t%s (from $GOROOT)", tried.goroot))
		} else {
			paths = append(paths, "\t($GOROOT not set)")
		}
		var i int
		var format = "\t%s (from $GOPATH)"
		for ; i < len(tried.gopath); i++ {
			if i > 0 {
				format = "\t%s"
			}
			paths = append(paths, fmt.Sprintf(format, tried.gopath[i]))
		}
		if i == 0 {
			paths = append(paths, "\t($GOPATH not set)")
		}
		return p, fmt.Errorf("cannot find package %q in any of:\n%s", path, strings.Join(paths, "\n"))
	}

Found:
	if p.Root != "" {
		if p.Goroot {
			p.SrcRoot = ctxt.joinPath(p.Root, "src", "pkg")
		} else {
			p.SrcRoot = ctxt.joinPath(p.Root, "src")
		}
		p.PkgRoot = ctxt.joinPath(p.Root, "pkg")
		p.BinDir = ctxt.joinPath(p.Root, "bin")
		if pkga != "" {
			p.PkgObj = ctxt.joinPath(p.Root, pkga)
		}
	}

	if mode&FindOnly != 0 {
		return p, pkgerr
	}
	if binaryOnly && (mode&AllowBinary) != 0 {
		return p, pkgerr
	}

	dirs, err := ctxt.readDir(p.Dir)
	if err != nil {
		return p, err
	}

	var Sfiles []string // files with ".S" (capital S)
	var firstFile string
	imported := make(map[string][]token.Position)
	testImported := make(map[string][]token.Position)
	xTestImported := make(map[string][]token.Position)
	fset := token.NewFileSet()
	for _, d := range dirs {
		if d.IsDir() {
			continue
		}
		name := d.Name()
		if strings.HasPrefix(name, "_") ||
			strings.HasPrefix(name, ".") {
			continue
		}
		if !ctxt.UseAllFiles && !ctxt.goodOSArchFile(name) {
			continue
		}

		i := strings.LastIndex(name, ".")
		if i < 0 {
			i = len(name)
		}
		ext := name[i:]
		switch ext {
		case ".go", ".c", ".s", ".h", ".S", ".swig", ".swigcxx":
			// tentatively okay - read to make sure
		case ".syso":
			// binary objects to add to package archive
			// Likely of the form foo_windows.syso, but
			// the name was vetted above with goodOSArchFile.
			p.SysoFiles = append(p.SysoFiles, name)
			continue
		default:
			// skip
			continue
		}

		filename := ctxt.joinPath(p.Dir, name)
		f, err := ctxt.openFile(filename)
		if err != nil {
			return p, err
		}

		var data []byte
		if strings.HasSuffix(filename, ".go") {
			data, err = readImports(f, false)
		} else {
			data, err = readComments(f)
		}
		f.Close()
		if err != nil {
			return p, fmt.Errorf("read %s: %v", filename, err)
		}

		// Look for +build comments to accept or reject the file.
		if !ctxt.UseAllFiles && !ctxt.shouldBuild(data) {
			continue
		}

		// Going to save the file.  For non-Go files, can stop here.
		switch ext {
		case ".c":
			p.CFiles = append(p.CFiles, name)
			continue
		case ".h":
			p.HFiles = append(p.HFiles, name)
			continue
		case ".s":
			p.SFiles = append(p.SFiles, name)
			continue
		case ".S":
			Sfiles = append(Sfiles, name)
			continue
		case ".swig":
			p.SwigFiles = append(p.SwigFiles, name)
			continue
		case ".swigcxx":
			p.SwigCXXFiles = append(p.SwigCXXFiles, name)
			continue
		}

		pf, err := parser.ParseFile(fset, filename, data, parser.ImportsOnly|parser.ParseComments)
		if err != nil {
			return p, err
		}

		pkg := pf.Name.Name
		if pkg == "documentation" {
			continue
		}

		isTest := strings.HasSuffix(name, "_test.go")
		isXTest := false
		if isTest && strings.HasSuffix(pkg, "_test") {
			isXTest = true
			pkg = pkg[:len(pkg)-len("_test")]
		}

		if p.Name == "" {
			p.Name = pkg
			firstFile = name
		} else if pkg != p.Name {
			return p, fmt.Errorf("found packages %s (%s) and %s (%s) in %s", p.Name, firstFile, pkg, name, p.Dir)
		}
		if pf.Doc != nil && p.Doc == "" {
			p.Doc = doc.Synopsis(pf.Doc.Text())
		}

		// Record imports and information about cgo.
		isCgo := false
		for _, decl := range pf.Decls {
			d, ok := decl.(*ast.GenDecl)
			if !ok {
				continue
			}
			for _, dspec := range d.Specs {
				spec, ok := dspec.(*ast.ImportSpec)
				if !ok {
					continue
				}
				quoted := spec.Path.Value
				path, err := strconv.Unquote(quoted)
				if err != nil {
					log.Panicf("%s: parser returned invalid quoted string: <%s>", filename, quoted)
				}
				if isXTest {
					xTestImported[path] = append(xTestImported[path], fset.Position(spec.Pos()))
				} else if isTest {
					testImported[path] = append(testImported[path], fset.Position(spec.Pos()))
				} else {
					imported[path] = append(imported[path], fset.Position(spec.Pos()))
				}
				if path == "C" {
					if isTest {
						return p, fmt.Errorf("use of cgo in test %s not supported", filename)
					}
					cg := spec.Doc
					if cg == nil && len(d.Specs) == 1 {
						cg = d.Doc
					}
					if cg != nil {
						if err := ctxt.saveCgo(filename, p, cg); err != nil {
							return p, err
						}
					}
					isCgo = true
				}
			}
		}
		if isCgo {
			if ctxt.CgoEnabled {
				p.CgoFiles = append(p.CgoFiles, name)
			}
		} else if isXTest {
			p.XTestGoFiles = append(p.XTestGoFiles, name)
		} else if isTest {
			p.TestGoFiles = append(p.TestGoFiles, name)
		} else {
			p.GoFiles = append(p.GoFiles, name)
		}
	}
	if p.Name == "" {
		return p, &NoGoError{p.Dir}
	}

	p.Imports, p.ImportPos = cleanImports(imported)
	p.TestImports, p.TestImportPos = cleanImports(testImported)
	p.XTestImports, p.XTestImportPos = cleanImports(xTestImported)

	// add the .S files only if we are using cgo
	// (which means gcc will compile them).
	// The standard assemblers expect .s files.
	if len(p.CgoFiles) > 0 {
		p.SFiles = append(p.SFiles, Sfiles...)
		sort.Strings(p.SFiles)
	}

	return p, pkgerr
}

func cleanImports(m map[string][]token.Position) ([]string, map[string][]token.Position) {
	all := make([]string, 0, len(m))
	for path := range m {
		all = append(all, path)
	}
	sort.Strings(all)
	return all, m
}

// Import is shorthand for Default.Import.
func Import(path, srcDir string, mode ImportMode) (*Package, error) {
	return Default.Import(path, srcDir, mode)
}

// ImportDir is shorthand for Default.ImportDir.
func ImportDir(dir string, mode ImportMode) (*Package, error) {
	return Default.ImportDir(dir, mode)
}

var slashslash = []byte("//")

// shouldBuild reports whether it is okay to use this file,
// The rule is that in the file's leading run of // comments
// and blank lines, which must be followed by a blank line
// (to avoid including a Go package clause doc comment),
// lines beginning with '// +build' are taken as build directives.
//
// The file is accepted only if each such line lists something
// matching the file.  For example:
//
//	// +build windows linux
//
// marks the file as applicable only on Windows and Linux.
//
func (ctxt *Context) shouldBuild(content []byte) bool {
	// Pass 1. Identify leading run of // comments and blank lines,
	// which must be followed by a blank line.
	end := 0
	p := content
	for len(p) > 0 {
		line := p
		if i := bytes.IndexByte(line, '\n'); i >= 0 {
			line, p = line[:i], p[i+1:]
		} else {
			p = p[len(p):]
		}
		line = bytes.TrimSpace(line)
		if len(line) == 0 { // Blank line
			end = len(content) - len(p)
			continue
		}
		if !bytes.HasPrefix(line, slashslash) { // Not comment line
			break
		}
	}
	content = content[:end]

	// Pass 2.  Process each line in the run.
	p = content
	for len(p) > 0 {
		line := p
		if i := bytes.IndexByte(line, '\n'); i >= 0 {
			line, p = line[:i], p[i+1:]
		} else {
			p = p[len(p):]
		}
		line = bytes.TrimSpace(line)
		if bytes.HasPrefix(line, slashslash) {
			line = bytes.TrimSpace(line[len(slashslash):])
			if len(line) > 0 && line[0] == '+' {
				// Looks like a comment +line.
				f := strings.Fields(string(line))
				if f[0] == "+build" {
					ok := false
					for _, tok := range f[1:] {
						if ctxt.match(tok) {
							ok = true
							break
						}
					}
					if !ok {
						return false // this one doesn't match
					}
				}
			}
		}
	}
	return true // everything matches
}

// saveCgo saves the information from the #cgo lines in the import "C" comment.
// These lines set CFLAGS and LDFLAGS and pkg-config directives that affect
// the way cgo's C code is built.
//
// TODO(rsc): This duplicates code in cgo.
// Once the dust settles, remove this code from cgo.
func (ctxt *Context) saveCgo(filename string, di *Package, cg *ast.CommentGroup) error {
	text := cg.Text()
	for _, line := range strings.Split(text, "\n") {
		orig := line

		// Line is
		//	#cgo [GOOS/GOARCH...] LDFLAGS: stuff
		//
		line = strings.TrimSpace(line)
		if len(line) < 5 || line[:4] != "#cgo" || (line[4] != ' ' && line[4] != '\t') {
			continue
		}

		// Split at colon.
		line = strings.TrimSpace(line[4:])
		i := strings.Index(line, ":")
		if i < 0 {
			return fmt.Errorf("%s: invalid #cgo line: %s", filename, orig)
		}
		line, argstr := line[:i], line[i+1:]

		// Parse GOOS/GOARCH stuff.
		f := strings.Fields(line)
		if len(f) < 1 {
			return fmt.Errorf("%s: invalid #cgo line: %s", filename, orig)
		}

		cond, verb := f[:len(f)-1], f[len(f)-1]
		if len(cond) > 0 {
			ok := false
			for _, c := range cond {
				if ctxt.match(c) {
					ok = true
					break
				}
			}
			if !ok {
				continue
			}
		}

		args, err := splitQuoted(argstr)
		if err != nil {
			return fmt.Errorf("%s: invalid #cgo line: %s", filename, orig)
		}
		for _, arg := range args {
			if !safeName(arg) {
				return fmt.Errorf("%s: malformed #cgo argument: %s", filename, arg)
			}
		}

		switch verb {
		case "CFLAGS":
			di.CgoCFLAGS = append(di.CgoCFLAGS, args...)
		case "LDFLAGS":
			di.CgoLDFLAGS = append(di.CgoLDFLAGS, args...)
		case "pkg-config":
			di.CgoPkgConfig = append(di.CgoPkgConfig, args...)
		default:
			return fmt.Errorf("%s: invalid #cgo verb: %s", filename, orig)
		}
	}
	return nil
}

var safeBytes = []byte("+-.,/0123456789=ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz:")

func safeName(s string) bool {
	if s == "" {
		return false
	}
	for i := 0; i < len(s); i++ {
		if c := s[i]; c < 0x80 && bytes.IndexByte(safeBytes, c) < 0 {
			return false
		}
	}
	return true
}

// splitQuoted splits the string s around each instance of one or more consecutive
// white space characters while taking into account quotes and escaping, and
// returns an array of substrings of s or an empty list if s contains only white space.
// Single quotes and double quotes are recognized to prevent splitting within the
// quoted region, and are removed from the resulting substrings. If a quote in s
// isn't closed err will be set and r will have the unclosed argument as the
// last element.  The backslash is used for escaping.
//
// For example, the following string:
//
//     a b:"c d" 'e''f'  "g\""
//
// Would be parsed as:
//
//     []string{"a", "b:c d", "ef", `g"`}
//
func splitQuoted(s string) (r []string, err error) {
	var args []string
	arg := make([]rune, len(s))
	escaped := false
	quoted := false
	quote := '\x00'
	i := 0
	for _, rune := range s {
		switch {
		case escaped:
			escaped = false
		case rune == '\\':
			escaped = true
			continue
		case quote != '\x00':
			if rune == quote {
				quote = '\x00'
				continue
			}
		case rune == '"' || rune == '\'':
			quoted = true
			quote = rune
			continue
		case unicode.IsSpace(rune):
			if quoted || i > 0 {
				quoted = false
				args = append(args, string(arg[:i]))
				i = 0
			}
			continue
		}
		arg[i] = rune
		i++
	}
	if quoted || i > 0 {
		args = append(args, string(arg[:i]))
	}
	if quote != 0 {
		err = errors.New("unclosed quote")
	} else if escaped {
		err = errors.New("unfinished escaping")
	}
	return args, err
}

// match returns true if the name is one of:
//
//	$GOOS
//	$GOARCH
//	cgo (if cgo is enabled)
//	!cgo (if cgo is disabled)
//	ctxt.Compiler
//	!ctxt.Compiler
//	tag (if tag is listed in ctxt.BuildTags)
//	!tag (if tag is not listed in ctxt.BuildTags)
//	a comma-separated list of any of these
//
func (ctxt *Context) match(name string) bool {
	if name == "" {
		return false
	}
	if i := strings.Index(name, ","); i >= 0 {
		// comma-separated list
		return ctxt.match(name[:i]) && ctxt.match(name[i+1:])
	}
	if strings.HasPrefix(name, "!!") { // bad syntax, reject always
		return false
	}
	if strings.HasPrefix(name, "!") { // negation
		return len(name) > 1 && !ctxt.match(name[1:])
	}

	// Tags must be letters, digits, underscores.
	// Unlike in Go identifiers, all digits are fine (e.g., "386").
	for _, c := range name {
		if !unicode.IsLetter(c) && !unicode.IsDigit(c) && c != '_' {
			return false
		}
	}

	// special tags
	if ctxt.CgoEnabled && name == "cgo" {
		return true
	}
	if name == ctxt.GOOS || name == ctxt.GOARCH || name == ctxt.Compiler {
		return true
	}

	// other tags
	for _, tag := range ctxt.BuildTags {
		if tag == name {
			return true
		}
	}

	return false
}

// goodOSArchFile returns false if the name contains a $GOOS or $GOARCH
// suffix which does not match the current system.
// The recognized name formats are:
//
//     name_$(GOOS).*
//     name_$(GOARCH).*
//     name_$(GOOS)_$(GOARCH).*
//     name_$(GOOS)_test.*
//     name_$(GOARCH)_test.*
//     name_$(GOOS)_$(GOARCH)_test.*
//
func (ctxt *Context) goodOSArchFile(name string) bool {
	if dot := strings.Index(name, "."); dot != -1 {
		name = name[:dot]
	}
	l := strings.Split(name, "_")
	if n := len(l); n > 0 && l[n-1] == "test" {
		l = l[:n-1]
	}
	n := len(l)
	if n >= 2 && knownOS[l[n-2]] && knownArch[l[n-1]] {
		return l[n-2] == ctxt.GOOS && l[n-1] == ctxt.GOARCH
	}
	if n >= 1 && knownOS[l[n-1]] {
		return l[n-1] == ctxt.GOOS
	}
	if n >= 1 && knownArch[l[n-1]] {
		return l[n-1] == ctxt.GOARCH
	}
	return true
}

var knownOS = make(map[string]bool)
var knownArch = make(map[string]bool)

func init() {
	for _, v := range strings.Fields(goosList) {
		knownOS[v] = true
	}
	for _, v := range strings.Fields(goarchList) {
		knownArch[v] = true
	}
}

// ToolDir is the directory containing build tools.
var ToolDir = filepath.Join(runtime.GOROOT(), "pkg/tool/"+runtime.GOOS+"_"+runtime.GOARCH)

// IsLocalImport reports whether the import path is
// a local import path, like ".", "..", "./foo", or "../foo".
func IsLocalImport(path string) bool {
	return path == "." || path == ".." ||
		strings.HasPrefix(path, "./") || strings.HasPrefix(path, "../")
}

// ArchChar returns the architecture character for the given goarch.
// For example, ArchChar("amd64") returns "6".
func ArchChar(goarch string) (string, error) {
	switch goarch {
	case "386":
		return "8", nil
	case "amd64":
		return "6", nil
	case "arm":
		return "5", nil
	}
	return "", errors.New("unsupported GOARCH " + goarch)
}
