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
	"io/ioutil"
	"log"
	"os"
	"path"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"unicode"
)

// A Context specifies the supporting context for a build.
type Context struct {
	GOARCH string // target architecture
	GOOS   string // target operating system
	// TODO(rsc,adg): GOPATH

	// By default, ScanDir uses the operating system's
	// file system calls to read directories and files.
	// Callers can override those calls to provide other
	// ways to read data by setting ReadDir and ReadFile.
	// ScanDir does not make any assumptions about the
	// format of the strings dir and file: they can be
	// slash-separated, backslash-separated, even URLs.

	// ReadDir returns a slice of os.FileInfo, sorted by Name,
	// describing the content of the named directory.
	// The dir argument is the argument to ScanDir.
	// If ReadDir is nil, ScanDir uses io.ReadDir.
	ReadDir func(dir string) (fi []os.FileInfo, err error)

	// ReadFile returns the content of the file named file
	// in the directory named dir.  The dir argument is the
	// argument to ScanDir, and the file argument is the
	// Name field from an os.FileInfo returned by ReadDir.
	// The returned path is the full name of the file, to be
	// used in error messages.
	//
	// If ReadFile is nil, ScanDir uses filepath.Join(dir, file)
	// as the path and ioutil.ReadFile to read the data.
	ReadFile func(dir, file string) (path string, content []byte, err error)
}

func (ctxt *Context) readDir(dir string) ([]os.FileInfo, error) {
	if f := ctxt.ReadDir; f != nil {
		return f(dir)
	}
	return ioutil.ReadDir(dir)
}

func (ctxt *Context) readFile(dir, file string) (string, []byte, error) {
	if f := ctxt.ReadFile; f != nil {
		return f(dir, file)
	}
	p := filepath.Join(dir, file)
	content, err := ioutil.ReadFile(p)
	return p, content, err
}

// The DefaultContext is the default Context for builds.
// It uses the GOARCH and GOOS environment variables
// if set, or else the compiled code's GOARCH and GOOS.
var DefaultContext = Context{
	GOARCH: envOr("GOARCH", runtime.GOARCH),
	GOOS:   envOr("GOOS", runtime.GOOS),
}

func envOr(name, def string) string {
	s := os.Getenv(name)
	if s == "" {
		return def
	}
	return s
}

type DirInfo struct {
	Package        string            // Name of package in dir
	PackageComment *ast.CommentGroup // Package comments from GoFiles
	ImportPath     string            // Import path of package in dir
	Imports        []string          // All packages imported by GoFiles

	// Source files
	GoFiles  []string // .go files in dir (excluding CgoFiles)
	CFiles   []string // .c files in dir
	SFiles   []string // .s files in dir
	CgoFiles []string // .go files that import "C"

	// Cgo directives
	CgoPkgConfig []string // Cgo pkg-config directives
	CgoCFLAGS    []string // Cgo CFLAGS directives
	CgoLDFLAGS   []string // Cgo LDFLAGS directives

	// Test information
	TestGoFiles  []string // _test.go files in package
	XTestGoFiles []string // _test.go files outside package
	TestImports  []string // All packages imported by (X)TestGoFiles
}

func (d *DirInfo) IsCommand() bool {
	// TODO(rsc): This is at least a little bogus.
	return d.Package == "main"
}

// ScanDir calls DefaultContext.ScanDir.
func ScanDir(dir string) (info *DirInfo, err error) {
	return DefaultContext.ScanDir(dir)
}

// ScanDir returns a structure with details about the Go content found
// in the given directory. The file lists exclude:
//
//	- files in package main (unless no other package is found)
//	- files in package documentation
//	- files ending in _test.go
//	- files starting with _ or .
//
func (ctxt *Context) ScanDir(dir string) (info *DirInfo, err error) {
	dirs, err := ctxt.readDir(dir)
	if err != nil {
		return nil, err
	}

	var di DirInfo
	imported := make(map[string]bool)
	testImported := make(map[string]bool)
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
		if !ctxt.goodOSArchFile(name) {
			continue
		}

		ext := path.Ext(name)
		switch ext {
		case ".go", ".c", ".s":
			// tentatively okay
		default:
			// skip
			continue
		}

		// Look for +build comments to accept or reject the file.
		filename, data, err := ctxt.readFile(dir, name)
		if err != nil {
			return nil, err
		}
		if !ctxt.shouldBuild(data) {
			continue
		}

		// Going to save the file.  For non-Go files, can stop here.
		switch ext {
		case ".c":
			di.CFiles = append(di.CFiles, name)
			continue
		case ".s":
			di.SFiles = append(di.SFiles, name)
			continue
		}

		pf, err := parser.ParseFile(fset, filename, data, parser.ImportsOnly|parser.ParseComments)
		if err != nil {
			return nil, err
		}

		pkg := string(pf.Name.Name)
		if pkg == "main" && di.Package != "" && di.Package != "main" {
			continue
		}
		if pkg == "documentation" {
			continue
		}

		isTest := strings.HasSuffix(name, "_test.go")
		if isTest && strings.HasSuffix(pkg, "_test") {
			pkg = pkg[:len(pkg)-len("_test")]
		}

		if pkg != di.Package && di.Package == "main" {
			// Found non-main package but was recording
			// information about package main.  Reset.
			di = DirInfo{}
		}
		if di.Package == "" {
			di.Package = pkg
		} else if pkg != di.Package {
			return nil, fmt.Errorf("%s: found packages %s and %s", dir, pkg, di.Package)
		}
		if pf.Doc != nil {
			if di.PackageComment != nil {
				di.PackageComment.List = append(di.PackageComment.List, pf.Doc.List...)
			} else {
				di.PackageComment = pf.Doc
			}
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
				quoted := string(spec.Path.Value)
				path, err := strconv.Unquote(quoted)
				if err != nil {
					log.Panicf("%s: parser returned invalid quoted string: <%s>", filename, quoted)
				}
				if isTest {
					testImported[path] = true
				} else {
					imported[path] = true
				}
				if path == "C" {
					if isTest {
						return nil, fmt.Errorf("%s: use of cgo in test not supported", filename)
					}
					cg := spec.Doc
					if cg == nil && len(d.Specs) == 1 {
						cg = d.Doc
					}
					if cg != nil {
						if err := ctxt.saveCgo(filename, &di, cg); err != nil {
							return nil, err
						}
					}
					isCgo = true
				}
			}
		}
		if isCgo {
			di.CgoFiles = append(di.CgoFiles, name)
		} else if isTest {
			if pkg == string(pf.Name.Name) {
				di.TestGoFiles = append(di.TestGoFiles, name)
			} else {
				di.XTestGoFiles = append(di.XTestGoFiles, name)
			}
		} else {
			di.GoFiles = append(di.GoFiles, name)
		}
	}
	if di.Package == "" {
		return nil, fmt.Errorf("%s: no Go source files", dir)
	}
	di.Imports = make([]string, len(imported))
	i := 0
	for p := range imported {
		di.Imports[i] = p
		i++
	}
	di.TestImports = make([]string, len(testImported))
	i = 0
	for p := range testImported {
		di.TestImports[i] = p
		i++
	}
	// File name lists are sorted because ReadDir sorts.
	sort.Strings(di.Imports)
	sort.Strings(di.TestImports)
	return &di, nil
}

var slashslash = []byte("//")
var plusBuild = []byte("+build")

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
			end = cap(content) - cap(line) // &line[0] - &content[0]
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
						if ctxt.matchOSArch(tok) {
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
func (ctxt *Context) saveCgo(filename string, di *DirInfo, cg *ast.CommentGroup) error {
	text := doc.CommentText(cg)
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
				if ctxt.matchOSArch(c) {
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

var safeBytes = []byte("+-.,/0123456789=ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz")

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

// matchOSArch returns true if the name is one of:
//
//	$GOOS
//	$GOARCH
//	$GOOS/$GOARCH
//
func (ctxt *Context) matchOSArch(name string) bool {
	if name == ctxt.GOOS || name == ctxt.GOARCH {
		return true
	}
	i := strings.Index(name, "/")
	return i >= 0 && name[:i] == ctxt.GOOS && name[i+1:] == ctxt.GOARCH
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
