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
	"path"
)

func (b *Builder) buildPackages(workpath string, hash string) os.Error {
	pkgs, err := packages()
	if err != nil {
		return err
	}
	for _, p := range pkgs {
		goroot := path.Join(workpath, "go")
		goinstall := path.Join(goroot, "bin", "goinstall")
		envv := append(b.envv(), "GOROOT="+goroot)

		// goinstall
		buildLog, code, err := runLog(envv, "", goroot, goinstall, p)
		if err != nil {
			log.Printf("goinstall %v: %v", p, err)
			continue
		}
		built := code != 0

		// get doc comment from package source
		info, err := packageComment(p, path.Join(goroot, "pkg", p))
		if err != nil {
			log.Printf("goinstall %v: %v", p, err)
		}

		// update dashboard with build state + info
		err = b.updatePackage(p, built, buildLog, info, hash)
		if err != nil {
			log.Printf("updatePackage %v: %v", p, err)
		}
	}
	return nil
}

func packageComment(pkg, pkgpath string) (info string, err os.Error) {
	fset := token.NewFileSet()
	pkgs, err := parser.ParseDir(fset, pkgpath, nil, parser.PackageClauseOnly|parser.ParseComments)
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
	return
}
