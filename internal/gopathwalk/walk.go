// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package gopathwalk is like filepath.Walk but specialized for finding Go
// packages, particularly in $GOPATH and $GOROOT.
package gopathwalk

import (
	"bufio"
	"bytes"
	"fmt"
	"go/build"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"strings"

	"golang.org/x/tools/internal/fastwalk"
)

// Options controls the behavior of a Walk call.
type Options struct {
	Debug          bool // Enable debug logging
	ModulesEnabled bool // Search module caches. Also disables legacy goimports ignore rules.
}

// RootType indicates the type of a Root.
type RootType int

const (
	RootUnknown RootType = iota
	RootGOROOT
	RootGOPATH
	RootCurrentModule
	RootModuleCache
	RootOther
)

// A Root is a starting point for a Walk.
type Root struct {
	Path string
	Type RootType
}

// SrcDirsRoots returns the roots from build.Default.SrcDirs(). Not modules-compatible.
func SrcDirsRoots(ctx *build.Context) []Root {
	var roots []Root
	roots = append(roots, Root{filepath.Join(ctx.GOROOT, "src"), RootGOROOT})
	for _, p := range filepath.SplitList(ctx.GOPATH) {
		roots = append(roots, Root{filepath.Join(p, "src"), RootGOPATH})
	}
	return roots
}

// Walk walks Go source directories ($GOROOT, $GOPATH, etc) to find packages.
// For each package found, add will be called (concurrently) with the absolute
// paths of the containing source directory and the package directory.
// add will be called concurrently.
func Walk(roots []Root, add func(root Root, dir string), opts Options) {
	for _, root := range roots {
		walkDir(root, add, opts)
	}
}

func walkDir(root Root, add func(Root, string), opts Options) {
	if _, err := os.Stat(root.Path); os.IsNotExist(err) {
		if opts.Debug {
			log.Printf("skipping nonexistent directory: %v", root.Path)
		}
		return
	}
	if opts.Debug {
		log.Printf("scanning %s", root.Path)
	}
	w := &walker{
		root: root,
		add:  add,
		opts: opts,
	}
	w.init()
	if err := fastwalk.Walk(root.Path, w.walk); err != nil {
		log.Printf("gopathwalk: scanning directory %v: %v", root.Path, err)
	}

	if opts.Debug {
		log.Printf("scanned %s", root.Path)
	}
}

// walker is the callback for fastwalk.Walk.
type walker struct {
	root Root               // The source directory to scan.
	add  func(Root, string) // The callback that will be invoked for every possible Go package dir.
	opts Options            // Options passed to Walk by the user.

	ignoredDirs []os.FileInfo // The ignored directories, loaded from .goimportsignore files.
}

// init initializes the walker based on its Options.
func (w *walker) init() {
	var ignoredPaths []string
	if w.root.Type == RootModuleCache {
		ignoredPaths = []string{"cache"}
	}
	if !w.opts.ModulesEnabled && w.root.Type == RootGOPATH {
		ignoredPaths = w.getIgnoredDirs(w.root.Path)
		ignoredPaths = append(ignoredPaths, "v", "mod")
	}

	for _, p := range ignoredPaths {
		full := filepath.Join(w.root.Path, p)
		if fi, err := os.Stat(full); err == nil {
			w.ignoredDirs = append(w.ignoredDirs, fi)
			if w.opts.Debug {
				log.Printf("Directory added to ignore list: %s", full)
			}
		} else if w.opts.Debug {
			log.Printf("Error statting ignored directory: %v", err)
		}
	}
}

// getIgnoredDirs reads an optional config file at <path>/.goimportsignore
// of relative directories to ignore when scanning for go files.
// The provided path is one of the $GOPATH entries with "src" appended.
func (w *walker) getIgnoredDirs(path string) []string {
	file := filepath.Join(path, ".goimportsignore")
	slurp, err := ioutil.ReadFile(file)
	if w.opts.Debug {
		if err != nil {
			log.Print(err)
		} else {
			log.Printf("Read %s", file)
		}
	}
	if err != nil {
		return nil
	}

	var ignoredDirs []string
	bs := bufio.NewScanner(bytes.NewReader(slurp))
	for bs.Scan() {
		line := strings.TrimSpace(bs.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		ignoredDirs = append(ignoredDirs, line)
	}
	return ignoredDirs
}

func (w *walker) shouldSkipDir(fi os.FileInfo) bool {
	for _, ignoredDir := range w.ignoredDirs {
		if os.SameFile(fi, ignoredDir) {
			return true
		}
	}
	return false
}

func (w *walker) walk(path string, typ os.FileMode) error {
	dir := filepath.Dir(path)
	if typ.IsRegular() {
		if dir == w.root.Path && (w.root.Type == RootGOROOT || w.root.Type == RootGOPATH) {
			// Doesn't make sense to have regular files
			// directly in your $GOPATH/src or $GOROOT/src.
			return fastwalk.SkipFiles
		}
		if !strings.HasSuffix(path, ".go") {
			return nil
		}

		w.add(w.root, dir)
		return fastwalk.SkipFiles
	}
	if typ == os.ModeDir {
		base := filepath.Base(path)
		if base == "" || base[0] == '.' || base[0] == '_' ||
			base == "testdata" ||
			(w.root.Type == RootGOROOT && w.opts.ModulesEnabled && base == "vendor") ||
			(!w.opts.ModulesEnabled && base == "node_modules") {
			return filepath.SkipDir
		}
		fi, err := os.Lstat(path)
		if err == nil && w.shouldSkipDir(fi) {
			return filepath.SkipDir
		}
		return nil
	}
	if typ == os.ModeSymlink {
		base := filepath.Base(path)
		if strings.HasPrefix(base, ".#") {
			// Emacs noise.
			return nil
		}
		fi, err := os.Lstat(path)
		if err != nil {
			// Just ignore it.
			return nil
		}
		if w.shouldTraverse(dir, fi) {
			return fastwalk.TraverseLink
		}
	}
	return nil
}

// shouldTraverse reports whether the symlink fi, found in dir,
// should be followed.  It makes sure symlinks were never visited
// before to avoid symlink loops.
func (w *walker) shouldTraverse(dir string, fi os.FileInfo) bool {
	path := filepath.Join(dir, fi.Name())
	target, err := filepath.EvalSymlinks(path)
	if err != nil {
		return false
	}
	ts, err := os.Stat(target)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return false
	}
	if !ts.IsDir() {
		return false
	}
	if w.shouldSkipDir(ts) {
		return false
	}
	// Check for symlink loops by statting each directory component
	// and seeing if any are the same file as ts.
	for {
		parent := filepath.Dir(path)
		if parent == path {
			// Made it to the root without seeing a cycle.
			// Use this symlink.
			return true
		}
		parentInfo, err := os.Stat(parent)
		if err != nil {
			return false
		}
		if os.SameFile(ts, parentInfo) {
			// Cycle. Don't traverse.
			return false
		}
		path = parent
	}

}
