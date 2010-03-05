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
func domake(dir, pkg string, local bool) os.Error {
	if local {
		_, err := os.Stat(dir + "/Makefile")
		if err == nil {
			return run(dir, nil, gobin+"/gomake", "install")
		}
	}
	makefile, err := makeMakefile(dir, pkg)
	if err != nil {
		return err
	}
	return run(dir, makefile, gobin+"/gomake", "-f-", "install")
}

// makeMakefile computes the standard Makefile for the directory dir
// installing as package pkg.  It includes all *.go files in the directory
// except those in package main and those ending in _test.go.
func makeMakefile(dir, pkg string) ([]byte, os.Error) {
	files, _, err := goFiles(dir)
	if err != nil {
		return nil, err
	}

	var buf bytes.Buffer
	if err := makefileTemplate.Execute(&makedata{pkg, files}, &buf); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// makedata is the data type for the makefileTemplate.
type makedata struct {
	pkg   string   // package import path
	files []string // list of .go files
}

var makefileTemplate = template.MustParse(`
include $(GOROOT)/src/Make.$(GOARCH)

TARG={pkg}
GOFILES=\
{.repeated section files}
	{@}\
{.end}

include $(GOROOT)/src/Make.pkg
`,
	nil)
