// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Run "make install" to build package.

package main

import (
	"bytes"
	"errors"
	"go/build"
	"path" // use for import paths
	"strings"
	"text/template"
)

// domake builds the package in dir.
// domake generates a standard Makefile and passes it
// to make on standard input.
func domake(dir, pkg string, tree *build.Tree, isCmd bool) (err error) {
	makefile, err := makeMakefile(dir, pkg, tree, isCmd)
	if err != nil {
		return err
	}
	cmd := []string{"bash", "gomake", "-f-"}
	if *nuke {
		cmd = append(cmd, "nuke")
	} else if *clean {
		cmd = append(cmd, "clean")
	}
	cmd = append(cmd, "install")
	return run(dir, makefile, cmd...)
}

// makeMakefile computes the standard Makefile for the directory dir
// installing as package pkg.  It includes all *.go files in the directory
// except those in package main and those ending in _test.go.
func makeMakefile(dir, pkg string, tree *build.Tree, isCmd bool) ([]byte, error) {
	if !safeName(pkg) {
		return nil, errors.New("unsafe name: " + pkg)
	}
	targ := pkg
	targDir := tree.PkgDir()
	if isCmd {
		// use the last part of the package name for targ
		_, targ = path.Split(pkg)
		targDir = tree.BinDir()
	}
	dirInfo, err := build.ScanDir(dir)
	if err != nil {
		return nil, err
	}

	cgoFiles := dirInfo.CgoFiles
	isCgo := make(map[string]bool, len(cgoFiles))
	for _, file := range cgoFiles {
		if !safeName(file) {
			return nil, errors.New("bad name: " + file)
		}
		isCgo[file] = true
	}

	goFiles := make([]string, 0, len(dirInfo.GoFiles))
	for _, file := range dirInfo.GoFiles {
		if !safeName(file) {
			return nil, errors.New("unsafe name: " + file)
		}
		if !isCgo[file] {
			goFiles = append(goFiles, file)
		}
	}

	oFiles := make([]string, 0, len(dirInfo.CFiles)+len(dirInfo.SFiles))
	cgoOFiles := make([]string, 0, len(dirInfo.CFiles))
	for _, file := range dirInfo.CFiles {
		if !safeName(file) {
			return nil, errors.New("unsafe name: " + file)
		}
		// When cgo is in use, C files are compiled with gcc,
		// otherwise they're compiled with gc.
		if len(cgoFiles) > 0 {
			cgoOFiles = append(cgoOFiles, file[:len(file)-2]+".o")
		} else {
			oFiles = append(oFiles, file[:len(file)-2]+".$O")
		}
	}

	for _, file := range dirInfo.SFiles {
		if !safeName(file) {
			return nil, errors.New("unsafe name: " + file)
		}
		oFiles = append(oFiles, file[:len(file)-2]+".$O")
	}

	var imports []string
	for _, t := range build.Path {
		imports = append(imports, t.PkgDir())
	}

	var buf bytes.Buffer
	md := makedata{targ, targDir, "pkg", goFiles, oFiles, cgoFiles, cgoOFiles, imports}
	if isCmd {
		md.Type = "cmd"
	}
	if err := makefileTemplate.Execute(&buf, &md); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

var safeBytes = []byte("+-~./0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz")

func safeName(s string) bool {
	if s == "" {
		return false
	}
	if strings.Contains(s, "..") {
		return false
	}
	if s[0] == '~' {
		return false
	}
	for i := 0; i < len(s); i++ {
		if c := s[i]; c < 0x80 && bytes.IndexByte(safeBytes, c) < 0 {
			return false
		}
	}
	return true
}

// makedata is the data type for the makefileTemplate.
type makedata struct {
	Targ      string   // build target
	TargDir   string   // build target directory
	Type      string   // build type: "pkg" or "cmd"
	GoFiles   []string // list of non-cgo .go files
	OFiles    []string // list of .$O files
	CgoFiles  []string // list of cgo .go files
	CgoOFiles []string // list of cgo .o files, without extension
	Imports   []string // gc/ld import paths
}

var makefileTemplate = template.Must(template.New("Makefile").Parse(`
include $(GOROOT)/src/Make.inc

TARG={{.Targ}}
TARGDIR={{.TargDir}}

{{with .GoFiles}}
GOFILES=\
{{range .}}	{{.}}\
{{end}}

{{end}}
{{with .OFiles}}
OFILES=\
{{range .}}	{{.}}\
{{end}}

{{end}}
{{with .CgoFiles}}
CGOFILES=\
{{range .}}	{{.}}\
{{end}}

{{end}}
{{with .CgoOFiles}}
CGO_OFILES=\
{{range .}}	{{.}}\
{{end}}

{{end}}
GCIMPORTS={{range .Imports}}-I "{{.}}" {{end}}
LDIMPORTS={{range .Imports}}-L "{{.}}" {{end}}

include $(GOROOT)/src/Make.{{.Type}}
`))
