// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO: Dashboard upload

package main

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
)

var cmdGet = &Command{
	UsageLine: "get [-a] [-d] [-fix] [-n] [-p n] [-u] [-v] [-x] [packages]",
	Short:     "download and install packages and dependencies",
	Long: `
Get downloads and installs the packages named by the import paths,
along with their dependencies.

The -a, -n, -v, -x, and -p flags have the same meaning as in 'go build'
and 'go install'.  See 'go help install'.

The -d flag instructs get to stop after downloading the packages; that is,
it instructs get not to install the packages.

The -fix flag instructs get to run the fix tool on the downloaded packages
before resolving dependencies or building the code.

The -u flag instructs get to use the network to update the named packages
and their dependencies.  By default, get uses the network to check out
missing packages but does not use it to look for updates to existing packages.

TODO: Explain versions better.

For more about specifying packages, see 'go help packages'.

For more about how 'go get' finds source code to
download, see 'go help remote'.

See also: go build, go install, go clean.
	`,
}

var getD = cmdGet.Flag.Bool("d", false, "")
var getU = cmdGet.Flag.Bool("u", false, "")
var getFix = cmdGet.Flag.Bool("fix", false, "")

func init() {
	addBuildFlags(cmdGet)
	cmdGet.Run = runGet // break init loop
}

func runGet(cmd *Command, args []string) {
	// Phase 1.  Download/update.
	args = importPaths(args)
	var stk importStack
	for _, arg := range args {
		download(arg, &stk)
	}
	exitIfErrors()

	if *getD {
		// download only
		return
	}

	// Phase 2. Install.

	// Code we downloaded and all code that depends on it
	// needs to be evicted from the package cache so that
	// the information will be recomputed.  Instead of keeping
	// track of the reverse dependency information, evict
	// everything.
	for name := range packageCache {
		delete(packageCache, name)
	}

	runInstall(cmd, args)
}

// downloadCache records the import paths we have already
// considered during the download, to avoid duplicate work when
// there is more than one dependency sequence leading to
// a particular package.
var downloadCache = map[string]bool{}

// downloadRootCache records the version control repository
// root directories we have already considered during the download.
// For example, all the packages in the code.google.com/p/codesearch repo
// share the same root (the directory for that path), and we only need
// to run the hg commands to consider each repository once.
var downloadRootCache = map[string]bool{}

// download runs the download half of the get command
// for the package named by the argument.
func download(arg string, stk *importStack) {
	p := loadPackage(arg, stk)

	// There's nothing to do if this is a package in the standard library.
	if p.Standard {
		return
	}

	// Only process each package once.
	if downloadCache[arg] {
		return
	}
	downloadCache[arg] = true

	// Download if the package is missing, or update if we're using -u.
	if p.Dir == "" || *getU {
		// The actual download.
		stk.push(p.ImportPath)
		defer stk.pop()
		if err := downloadPackage(p); err != nil {
			errorf("%s", &PackageError{ImportStack: stk.copy(), Err: err.Error()})
			return
		}

		// Reread the package information from the updated files.
		p = reloadPackage(arg, stk)
		if p.Error != nil {
			errorf("%s", p.Error)
			return
		}
	}

	if *getFix {
		run(stringList(tool("fix"), relPaths(p.gofiles)))

		// The imports might have changed, so reload again.
		p = reloadPackage(arg, stk)
		if p.Error != nil {
			errorf("%s", p.Error)
			return
		}
	}

	// Process dependencies, now that we know what they are.
	for _, dep := range p.deps {
		download(dep.ImportPath, stk)
	}
}

// downloadPackage runs the create or download command
// to make the first copy of or update a copy of the given package.
func downloadPackage(p *Package) error {
	var (
		vcs            *vcsCmd
		repo, rootPath string
		err            error
	)
	if p.build.SrcRoot != "" {
		// Directory exists.  Look for checkout along path to src.
		vcs, rootPath, err = vcsForDir(p)
		repo = "<local>" // should be unused; make distinctive
	} else {
		// Analyze the import path to determine the version control system,
		// repository, and the import path for the root of the repository.
		rr, err := repoRootForImportPath(p.ImportPath)
		if err != nil {
			return err
		}
		vcs, repo, rootPath = rr.vcs, rr.repo, rr.root
	}

	if p.build.SrcRoot == "" {
		// Package not found.  Put in first directory of $GOPATH or else $GOROOT.
		if list := filepath.SplitList(buildContext.GOPATH); len(list) > 0 {
			p.build.SrcRoot = filepath.Join(list[0], "src")
			p.build.PkgRoot = filepath.Join(list[0], "pkg")
		} else {
			p.build.SrcRoot = filepath.Join(goroot, "src", "pkg")
			p.build.PkgRoot = filepath.Join(goroot, "pkg")
		}
	}
	root := filepath.Join(p.build.SrcRoot, rootPath)
	// If we've considered this repository already, don't do it again.
	if downloadRootCache[root] {
		return nil
	}
	downloadRootCache[root] = true

	if buildV {
		fmt.Fprintf(os.Stderr, "%s (download)\n", rootPath)
	}

	// Check that this is an appropriate place for the repo to be checked out.
	// The target directory must either not exist or have a repo checked out already.
	meta := filepath.Join(root, "."+vcs.cmd)
	st, err := os.Stat(meta)
	if err == nil && !st.IsDir() {
		return fmt.Errorf("%s exists but is not a directory", meta)
	}
	if err != nil {
		// Metadata directory does not exist.  Prepare to checkout new copy.
		// Some version control tools require the target directory not to exist.
		// We require that too, just to avoid stepping on existing work.
		if _, err := os.Stat(root); err == nil {
			return fmt.Errorf("%s exists but %s does not - stale checkout?", root, meta)
		}
		// Some version control tools require the parent of the target to exist.
		parent, _ := filepath.Split(root)
		if err := os.MkdirAll(parent, 0777); err != nil {
			return err
		}
		if err = vcs.create(root, repo); err != nil {
			return err
		}
	} else {
		// Metadata directory does exist; download incremental updates.
		if err = vcs.download(root); err != nil {
			return err
		}
	}

	if buildN {
		// Do not show tag sync in -n; it's noise more than anything,
		// and since we're not running commands, no tag will be found.
		// But avoid printing nothing.
		fmt.Fprintf(os.Stderr, "# cd %s; %s sync/update\n", root, vcs.cmd)
		return nil
	}

	// Select and sync to appropriate version of the repository.
	tags, err := vcs.tags(root)
	if err != nil {
		return err
	}
	vers := runtime.Version()
	if i := strings.Index(vers, " "); i >= 0 {
		vers = vers[:i]
	}
	if err := vcs.tagSync(root, selectTag(vers, tags)); err != nil {
		return err
	}

	return nil
}

// selectTag returns the closest matching tag for a given version.
// Closest means the latest one that is not after the current release.
// Version "release.rN" matches tags of the form "go.rN" (N being a decimal).
// Version "weekly.YYYY-MM-DD" matches tags like "go.weekly.YYYY-MM-DD".
func selectTag(goVersion string, tags []string) (match string) {
	const rPrefix = "release.r"
	if strings.HasPrefix(goVersion, rPrefix) {
		p := "go.r"
		v, err := strconv.ParseFloat(goVersion[len(rPrefix):], 64)
		if err != nil {
			return ""
		}
		var matchf float64
		for _, t := range tags {
			if !strings.HasPrefix(t, p) {
				continue
			}
			tf, err := strconv.ParseFloat(t[len(p):], 64)
			if err != nil {
				continue
			}
			if matchf < tf && tf <= v {
				match, matchf = t, tf
			}
		}
	}
	const wPrefix = "weekly."
	if strings.HasPrefix(goVersion, wPrefix) {
		p := "go.weekly."
		v := goVersion[len(wPrefix):]
		for _, t := range tags {
			if !strings.HasPrefix(t, p) {
				continue
			}
			if match < t && t[len(p):] <= v {
				match = t
			}
		}
	}
	return match
}
