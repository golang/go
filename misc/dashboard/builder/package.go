// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"errors"
	"fmt"
	"go/doc"
	"go/parser"
	"go/token"
	"log"
	"os"
	"path/filepath"
	"strings"
)

const MaxCommentLength = 500 // App Engine won't store more in a StringProperty.

func (b *Builder) buildPackages(workpath string, hash string) error {
	logdir := filepath.Join(*buildroot, "log")
	if err := os.Mkdir(logdir, 0755); err != nil {
		return err
	}
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
		buildLog, code, err := runLog(envv, "", goroot, goinstall, "-dashboard=false", p)
		if err != nil {
			log.Printf("goinstall %v: %v", p, err)
		}

		// get doc comment from package source
		var info string
		pkgPath := filepath.Join(goroot, "src", "pkg", p)
		if _, err := os.Stat(pkgPath); err == nil {
			info, err = packageComment(p, pkgPath)
			if err != nil {
				log.Printf("packageComment %v: %v", p, err)
			}
		}

		// update dashboard with build state + info
		err = b.updatePackage(p, code == 0, buildLog, info)
		if err != nil {
			log.Printf("updatePackage %v: %v", p, err)
		}

		if code == 0 {
			log.Println("Build succeeded:", p)
		} else {
			log.Println("Build failed:", p)
			fn := filepath.Join(logdir, strings.Replace(p, "/", "_", -1))
			if f, err := os.Create(fn); err != nil {
				log.Printf("creating %s: %v", fn, err)
			} else {
				fmt.Fprint(f, buildLog)
				f.Close()
			}
		}
	}
	return nil
}

func isGoFile(fi os.FileInfo) bool {
	return !fi.IsDir() && // exclude directories
		!strings.HasPrefix(fi.Name(), ".") && // ignore .files
		!strings.HasSuffix(fi.Name(), "_test.go") && // ignore tests
		filepath.Ext(fi.Name()) == ".go"
}

func packageComment(pkg, pkgpath string) (info string, err error) {
	fset := token.NewFileSet()
	pkgs, err := parser.ParseDir(fset, pkgpath, isGoFile, parser.PackageClauseOnly|parser.ParseComments)
	if err != nil {
		return
	}
	for name := range pkgs {
		if name == "main" {
			continue
		}
		pdoc := doc.NewPackageDoc(pkgs[name], pkg)
		if pdoc.Doc == "" {
			continue
		}
		if info != "" {
			return "", errors.New("multiple packages with docs")
		}
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
