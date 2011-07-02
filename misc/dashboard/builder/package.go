// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"go/doc"
	"go/parser"
	"go/token"
	"log"
	"os"
	"path/filepath"
	"strings"
)

const MaxCommentLength = 500 // App Engine won't store more in a StringProperty.

func (b *Builder) buildPackages(workpath string, hash string) os.Error {
	pkgs, err := packages()
	if err != nil {
		return err
	}
	for _, p := range pkgs {
		goroot := filepath.Join(workpath, "go")
		gobin := filepath.Join(goroot, "bin")
		goinstall := filepath.Join(gobin, "goinstall")
		envv := append(b.envv(), "GOROOT="+goroot)

		// add GOBIN to path
		for i, v := range envv {
			if strings.HasPrefix(v, "PATH=") {
				p := filepath.SplitList(v[5:])
				p = append([]string{gobin}, p...)
				s := strings.Join(p, string(filepath.ListSeparator))
				envv[i] = "PATH=" + s
			}
		}

		// goinstall
		buildLog, code, err := runLog(envv, "", goroot, goinstall, "-log=false", p)
		if err != nil {
			log.Printf("goinstall %v: %v", p, err)
		}

		// get doc comment from package source
		info, err := packageComment(p, filepath.Join(goroot, "src", "pkg", p))
		if err != nil {
			log.Printf("packageComment %v: %v", p, err)
		}

		// update dashboard with build state + info
		err = b.updatePackage(p, code == 0, buildLog, info)
		if err != nil {
			log.Printf("updatePackage %v: %v", p, err)
		}
	}
	return nil
}

func isGoFile(fi *os.FileInfo) bool {
	return fi.IsRegular() && // exclude directories
		!strings.HasPrefix(fi.Name, ".") && // ignore .files
		filepath.Ext(fi.Name) == ".go"
}

func packageComment(pkg, pkgpath string) (info string, err os.Error) {
	fset := token.NewFileSet()
	pkgs, err := parser.ParseDir(fset, pkgpath, isGoFile, parser.PackageClauseOnly|parser.ParseComments)
	if err != nil {
		return
	}
	for name := range pkgs {
		if name == "main" {
			continue
		}
		if info != "" {
			return "", os.NewError("multiple non-main package docs")
		}
		pdoc := doc.NewPackageDoc(pkgs[name], pkg)
		info = pdoc.Doc
	}
	// grab only first paragraph
	if parts := strings.SplitN(info, "\n\n", 2); len(parts) > 1 {
		info = parts[0]
	}
	// replace newlines with spaces
	info = strings.Replace(info, "\n", " ", -1)
	// truncate
	if len(info) > MaxCommentLength {
		info = info[:MaxCommentLength]
	}
	return
}
