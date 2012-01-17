// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package build

import (
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"runtime"
)

// Path is a validated list of Trees derived from $GOROOT and $GOPATH at init.
var Path []*Tree

// Tree describes a Go source tree, either $GOROOT or one from $GOPATH.
type Tree struct {
	Path   string
	Goroot bool
}

func newTree(p string) (*Tree, error) {
	if !filepath.IsAbs(p) {
		return nil, errors.New("must be absolute")
	}
	ep, err := filepath.EvalSymlinks(p)
	if err != nil {
		return nil, err
	}
	return &Tree{Path: ep}, nil
}

// SrcDir returns the tree's package source directory.
func (t *Tree) SrcDir() string {
	if t.Goroot {
		return filepath.Join(t.Path, "src", "pkg")
	}
	return filepath.Join(t.Path, "src")
}

// PkgDir returns the tree's package object directory.
func (t *Tree) PkgDir() string {
	goos, goarch := runtime.GOOS, runtime.GOARCH
	if e := os.Getenv("GOOS"); e != "" {
		goos = e
	}
	if e := os.Getenv("GOARCH"); e != "" {
		goarch = e
	}
	return filepath.Join(t.Path, "pkg", goos+"_"+goarch)
}

// BinDir returns the tree's binary executable directory.
func (t *Tree) BinDir() string {
	if t.Goroot {
		if gobin := os.Getenv("GOBIN"); gobin != "" {
			return filepath.Clean(gobin)
		}
	}
	return filepath.Join(t.Path, "bin")
}

// HasSrc returns whether the given package's
// source can be found inside this Tree.
func (t *Tree) HasSrc(pkg string) bool {
	fi, err := os.Stat(filepath.Join(t.SrcDir(), pkg))
	if err != nil {
		return false
	}
	return fi.IsDir()
}

// HasPkg returns whether the given package's
// object file can be found inside this Tree.
func (t *Tree) HasPkg(pkg string) bool {
	fi, err := os.Stat(filepath.Join(t.PkgDir(), pkg+".a"))
	if err != nil {
		return false
	}
	return !fi.IsDir()
	// TODO(adg): check object version is consistent
}

var (
	ErrNotFound     = errors.New("package could not be found locally")
	ErrTreeNotFound = errors.New("no valid GOROOT or GOPATH could be found")
)

// FindTree takes an import or filesystem path and returns the
// tree where the package source should be and the package import path.
func FindTree(path string) (tree *Tree, pkg string, err error) {
	if isLocalPath(path) {
		if path, err = filepath.Abs(path); err != nil {
			return
		}
		if path, err = filepath.EvalSymlinks(path); err != nil {
			return
		}
		for _, t := range Path {
			tpath := t.SrcDir() + string(filepath.Separator)
			if !filepath.HasPrefix(path, tpath) {
				continue
			}
			tree = t
			pkg = filepath.ToSlash(path[len(tpath):])
			return
		}
		err = fmt.Errorf("path %q not inside a GOPATH", path)
		return
	}
	tree = defaultTree
	pkg = filepath.ToSlash(path)
	for _, t := range Path {
		if t.HasSrc(pkg) {
			tree = t
			return
		}
	}
	if tree == nil {
		err = ErrTreeNotFound
	} else {
		err = ErrNotFound
	}
	return
}

// isLocalPath returns whether the given path is local (/foo ./foo ../foo . ..)
// Windows paths that starts with drive letter (c:\foo c:foo) are considered local.
func isLocalPath(s string) bool {
	const sep = string(filepath.Separator)
	return s == "." || s == ".." ||
		filepath.HasPrefix(s, sep) ||
		filepath.HasPrefix(s, "."+sep) || filepath.HasPrefix(s, ".."+sep) ||
		filepath.VolumeName(s) != ""
}

var (
	// argument lists used by the build's gc and ld methods
	gcImportArgs []string
	ldImportArgs []string

	// default tree for remote packages
	defaultTree *Tree
)

// set up Path: parse and validate GOROOT and GOPATH variables
func init() {
	root := runtime.GOROOT()
	t, err := newTree(root)
	if err != nil {
		log.Printf("invalid GOROOT %q: %v", root, err)
	} else {
		t.Goroot = true
		Path = []*Tree{t}
	}

Loop:
	for _, p := range filepath.SplitList(os.Getenv("GOPATH")) {
		if p == "" {
			continue
		}
		t, err := newTree(p)
		if err != nil {
			log.Printf("invalid GOPATH %q: %v", p, err)
			continue
		}

		// Check for dupes.
		// TODO(alexbrainman): make this correct under windows (case insensitive).
		for _, t2 := range Path {
			if t2.Path != t.Path {
				continue
			}
			if t2.Goroot {
				log.Printf("GOPATH is the same as GOROOT: %q", t.Path)
			} else {
				log.Printf("duplicate GOPATH entry: %q", t.Path)
			}
			continue Loop
		}

		Path = append(Path, t)
		gcImportArgs = append(gcImportArgs, "-I", t.PkgDir())
		ldImportArgs = append(ldImportArgs, "-L", t.PkgDir())

		// select first GOPATH entry as default
		if defaultTree == nil {
			defaultTree = t
		}
	}

	// use GOROOT if no valid GOPATH specified
	if defaultTree == nil && len(Path) > 0 {
		defaultTree = Path[0]
	}
}
