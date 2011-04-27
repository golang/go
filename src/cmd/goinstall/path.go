// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"log"
	"os"
	"path/filepath"
	"runtime"
)

var (
	gopath      []*pkgroot
	imports     []string
	defaultRoot *pkgroot // default root for remote packages
)

// set up gopath: parse and validate GOROOT and GOPATH variables
func init() {
	p, err := newPkgroot(root)
	if err != nil {
		log.Fatalf("Invalid GOROOT %q: %v", root, err)
	}
	p.goroot = true
	gopath = []*pkgroot{p}

	for _, p := range filepath.SplitList(os.Getenv("GOPATH")) {
		if p == "" {
			continue
		}
		r, err := newPkgroot(p)
		if err != nil {
			log.Printf("Invalid GOPATH %q: %v", p, err)
			continue
		}
		gopath = append(gopath, r)
		imports = append(imports, r.pkgDir())

		// select first GOPATH entry as default
		if defaultRoot == nil {
			defaultRoot = r
		}
	}

	// use GOROOT if no valid GOPATH specified
	if defaultRoot == nil {
		defaultRoot = gopath[0]
	}
}

type pkgroot struct {
	path   string
	goroot bool // TODO(adg): remove this once Go tree re-organized
}

func newPkgroot(p string) (*pkgroot, os.Error) {
	if !filepath.IsAbs(p) {
		return nil, os.NewError("must be absolute")
	}
	ep, err := filepath.EvalSymlinks(p)
	if err != nil {
		return nil, err
	}
	return &pkgroot{path: ep}, nil
}

func (r *pkgroot) srcDir() string {
	if r.goroot {
		return filepath.Join(r.path, "src", "pkg")
	}
	return filepath.Join(r.path, "src")
}

func (r *pkgroot) pkgDir() string {
	goos, goarch := runtime.GOOS, runtime.GOARCH
	if e := os.Getenv("GOOS"); e != "" {
		goos = e
	}
	if e := os.Getenv("GOARCH"); e != "" {
		goarch = e
	}
	return filepath.Join(r.path, "pkg", goos+"_"+goarch)
}

func (r *pkgroot) binDir() string {
	return filepath.Join(r.path, "bin")
}

func (r *pkgroot) hasSrcDir(name string) bool {
	fi, err := os.Stat(filepath.Join(r.srcDir(), name))
	if err != nil {
		return false
	}
	return fi.IsDirectory()
}

func (r *pkgroot) hasPkg(name string) bool {
	fi, err := os.Stat(filepath.Join(r.pkgDir(), name+".a"))
	if err != nil {
		return false
	}
	return fi.IsRegular()
	// TODO(adg): check object version is consistent
}

// findPkgroot searches each of the gopath roots
// for the source code for the given import path.
func findPkgroot(importPath string) *pkgroot {
	for _, r := range gopath {
		if r.hasSrcDir(importPath) {
			return r
		}
	}
	return defaultRoot
}
