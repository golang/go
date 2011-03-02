// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Run "make install" to build package.

package main

import (
	"bytes"
	"os"
	"template"
)

// domake builds the package in dir.
// If local is false, the package was copied from an external system.
// For non-local packages or packages without Makefiles,
// domake generates a standard Makefile and passes it
// to make on standard input.
func domake(dir, pkg string, local bool) (err os.Error) {
	needMakefile := true
	if local {
		_, err := os.Stat(dir + "/Makefile")
		if err == nil {
			needMakefile = false
		}
	}
	cmd := []string{"gomake"}
	var makefile []byte
	if needMakefile {
		if makefile, err = makeMakefile(dir, pkg); err != nil {
			return err
		}
		cmd = append(cmd, "-f-")
	}
	if *clean {
		cmd = append(cmd, "clean")
	}
	cmd = append(cmd, "install")
	return run(dir, makefile, cmd...)
}

// makeMakefile computes the standard Makefile for the directory dir
// installing as package pkg.  It includes all *.go files in the directory
// except those in package main and those ending in _test.go.
func makeMakefile(dir, pkg string) ([]byte, os.Error) {
	if !safeName(pkg) {
		return nil, os.ErrorString("unsafe name: " + pkg)
	}
	dirInfo, err := scanDir(dir, false)
	if err != nil {
		return nil, err
	}

	if len(dirInfo.cgoFiles) == 0 && len(dirInfo.cFiles) > 0 {
		// When using cgo, .c files are compiled with gcc.  Without cgo,
		// they may be intended for 6c.  Just error out for now.
		return nil, os.ErrorString("C files found in non-cgo package")
	}

	cgoFiles := dirInfo.cgoFiles
	isCgo := make(map[string]bool, len(cgoFiles))
	for _, file := range cgoFiles {
		if !safeName(file) {
			return nil, os.ErrorString("bad name: " + file)
		}
		isCgo[file] = true
	}

	oFiles := make([]string, 0, len(dirInfo.cFiles))
	for _, file := range dirInfo.cFiles {
		if !safeName(file) {
			return nil, os.ErrorString("unsafe name: " + file)
		}
		oFiles = append(oFiles, file[:len(file)-2]+".o")
	}

	goFiles := make([]string, 0, len(dirInfo.goFiles))
	for _, file := range dirInfo.goFiles {
		if !safeName(file) {
			return nil, os.ErrorString("unsafe name: " + file)
		}
		if !isCgo[file] {
			goFiles = append(goFiles, file)
		}
	}

	var buf bytes.Buffer
	md := makedata{pkg, goFiles, cgoFiles, oFiles}
	if err := makefileTemplate.Execute(&buf, &md); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

var safeBytes = []byte("+-./0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz")

func safeName(s string) bool {
	for i := 0; i < len(s); i++ {
		if c := s[i]; c < 0x80 && bytes.IndexByte(safeBytes, c) < 0 {
			return false
		}
	}
	return true
}

// makedata is the data type for the makefileTemplate.
type makedata struct {
	Pkg      string   // package import path
	GoFiles  []string // list of non-cgo .go files
	CgoFiles []string // list of cgo .go files
	OFiles   []string // list of ofiles for cgo
}

var makefileTemplate = template.MustParse(`
include $(GOROOT)/src/Make.inc

TARG={Pkg}

{.section GoFiles}
GOFILES=\
{.repeated section GoFiles}
	{@}\
{.end}

{.end}
{.section CgoFiles}
CGOFILES=\
{.repeated section CgoFiles}
	{@}\
{.end}

{.end}
{.section OFiles}
CGO_OFILES=\
{.repeated section OFiles}
	{@}\
{.end}

{.end}
include $(GOROOT)/src/Make.pkg
`,
	nil)
