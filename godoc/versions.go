// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file caches information about which standard library types, methods,
// and functions appeared in what version of Go

package godoc

import (
	"bufio"
	"go/build"
	"log"
	"os"
	"path/filepath"
	"strings"
	"unicode"
)

// apiVersions is a map of packages to information about those packages'
// symbols and when they were added to Go.
//
// Only things added after Go1 are tracked. Version strings are of the
// form "1.1", "1.2", etc.
type apiVersions map[string]pkgAPIVersions // keyed by Go package ("net/http")

// pkgAPIVersions contains information about which version of Go added
// certain package symbols.
//
// Only things added after Go1 are tracked. Version strings are of the
// form "1.1", "1.2", etc.
type pkgAPIVersions struct {
	typeSince   map[string]string            // "Server" -> "1.7"
	methodSince map[string]map[string]string // "*Server" ->"Shutdown"->1.8
	funcSince   map[string]string            // "NewServer" -> "1.7"
	fieldSince  map[string]map[string]string // "ClientTrace" -> "Got1xxResponse" -> "1.11"
}

// sinceVersionFunc returns a string (such as "1.7") specifying which Go
// version introduced a symbol, unless it was introduced in Go1, in
// which case it returns the empty string.
//
// The kind is one of "type", "method", or "func".
//
// The receiver is only used for "methods" and specifies the receiver type,
// such as "*Server".
//
// The name is the symbol name ("Server") and the pkg is the package
// ("net/http").
func (v apiVersions) sinceVersionFunc(kind, receiver, name, pkg string) string {
	pv := v[pkg]
	switch kind {
	case "func":
		return pv.funcSince[name]
	case "type":
		return pv.typeSince[name]
	case "method":
		return pv.methodSince[receiver][name]
	}
	return ""
}

// versionedRow represents an API feature, a parsed line of a
// $GOROOT/api/go.*txt file.
type versionedRow struct {
	pkg        string // "net/http"
	kind       string // "type", "func", "method", "field" TODO: "const", "var"
	recv       string // for methods, the receiver type ("Server", "*Server")
	name       string // name of type, (struct) field, func, method
	structName string // for struct fields, the outer struct name
}

// versionParser parses $GOROOT/api/go*.txt files and stores them in in its rows field.
type versionParser struct {
	res apiVersions // initialized lazily
}

func (vp *versionParser) parseFile(name string) error {
	base := filepath.Base(name)
	ver := strings.TrimPrefix(strings.TrimSuffix(base, ".txt"), "go")
	if ver == "1" {
		return nil
	}
	f, err := os.Open(name)
	if err != nil {
		return err
	}
	defer f.Close()

	sc := bufio.NewScanner(f)
	for sc.Scan() {
		row, ok := parseRow(sc.Text())
		if !ok {
			continue
		}
		if vp.res == nil {
			vp.res = make(apiVersions)
		}
		pkgi, ok := vp.res[row.pkg]
		if !ok {
			pkgi = pkgAPIVersions{
				typeSince:   make(map[string]string),
				methodSince: make(map[string]map[string]string),
				funcSince:   make(map[string]string),
				fieldSince:  make(map[string]map[string]string),
			}
			vp.res[row.pkg] = pkgi
		}
		switch row.kind {
		case "func":
			pkgi.funcSince[row.name] = ver
		case "type":
			pkgi.typeSince[row.name] = ver
		case "method":
			if _, ok := pkgi.methodSince[row.recv]; !ok {
				pkgi.methodSince[row.recv] = make(map[string]string)
			}
			pkgi.methodSince[row.recv][row.name] = ver
		case "field":
			if _, ok := pkgi.fieldSince[row.structName]; !ok {
				pkgi.fieldSince[row.structName] = make(map[string]string)
			}
			pkgi.fieldSince[row.structName][row.name] = ver
		}
	}
	return sc.Err()
}

func parseRow(s string) (vr versionedRow, ok bool) {
	if !strings.HasPrefix(s, "pkg ") {
		// Skip comments, blank lines, etc.
		return
	}
	rest := s[len("pkg "):]
	endPkg := strings.IndexFunc(rest, func(r rune) bool { return !(unicode.IsLetter(r) || r == '/' || unicode.IsDigit(r)) })
	if endPkg == -1 {
		return
	}
	vr.pkg, rest = rest[:endPkg], rest[endPkg:]
	if !strings.HasPrefix(rest, ", ") {
		// If the part after the pkg name isn't ", ", then it's a OS/ARCH-dependent line of the form:
		//   pkg syscall (darwin-amd64), const ImplementsGetwd = false
		// We skip those for now.
		return
	}
	rest = rest[len(", "):]

	switch {
	case strings.HasPrefix(rest, "type "):
		rest = rest[len("type "):]
		sp := strings.IndexByte(rest, ' ')
		if sp == -1 {
			return
		}
		vr.name, rest = rest[:sp], rest[sp+1:]
		if !strings.HasPrefix(rest, "struct, ") {
			vr.kind = "type"
			return vr, true
		}
		rest = rest[len("struct, "):]
		if i := strings.IndexByte(rest, ' '); i != -1 {
			vr.kind = "field"
			vr.structName = vr.name
			vr.name = rest[:i]
			return vr, true
		}
	case strings.HasPrefix(rest, "func "):
		vr.kind = "func"
		rest = rest[len("func "):]
		if i := strings.IndexByte(rest, '('); i != -1 {
			vr.name = rest[:i]
			return vr, true
		}
	case strings.HasPrefix(rest, "method "): // "method (*File) SetModTime(time.Time)"
		vr.kind = "method"
		rest = rest[len("method "):] // "(*File) SetModTime(time.Time)"
		sp := strings.IndexByte(rest, ' ')
		if sp == -1 {
			return
		}
		vr.recv = strings.Trim(rest[:sp], "()") // "*File"
		rest = rest[sp+1:]                      // SetMode(os.FileMode)
		paren := strings.IndexByte(rest, '(')
		if paren == -1 {
			return
		}
		vr.name = rest[:paren]
		return vr, true
	}
	return // TODO: handle more cases
}

// InitVersionInfo parses the $GOROOT/api/go*.txt API definition files to discover
// which API features were added in which Go releases.
func (c *Corpus) InitVersionInfo() {
	var err error
	c.pkgAPIInfo, err = parsePackageAPIInfo()
	if err != nil {
		// TODO: consider making this fatal, after the Go 1.11 cycle.
		log.Printf("godoc: error parsing API version files: %v", err)
	}
}

func parsePackageAPIInfo() (apiVersions, error) {
	var apiGlob string
	if os.Getenv("GOROOT") == "" {
		apiGlob = filepath.Join(build.Default.GOROOT, "api", "go*.txt")
	} else {
		apiGlob = filepath.Join(os.Getenv("GOROOT"), "api", "go*.txt")
	}

	files, err := filepath.Glob(apiGlob)
	if err != nil {
		return nil, err
	}

	vp := new(versionParser)
	for _, f := range files {
		if err := vp.parseFile(f); err != nil {
			return nil, err
		}
	}
	return vp.res, nil
}
