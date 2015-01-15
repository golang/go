// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"go/build"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
)

var cmdGet = &Command{
	UsageLine: "get [-d] [-f] [-fix] [-t] [-u] [build flags] [packages]",
	Short:     "download and install packages and dependencies",
	Long: `
Get downloads and installs the packages named by the import paths,
along with their dependencies.

The -d flag instructs get to stop after downloading the packages; that is,
it instructs get not to install the packages.

The -f flag, valid only when -u is set, forces get -u not to verify that
each package has been checked out from the source control repository
implied by its import path. This can be useful if the source is a local fork
of the original.

The -fix flag instructs get to run the fix tool on the downloaded packages
before resolving dependencies or building the code.

The -t flag instructs get to also download the packages required to build
the tests for the specified packages.

The -u flag instructs get to use the network to update the named packages
and their dependencies.  By default, get uses the network to check out
missing packages but does not use it to look for updates to existing packages.

Get also accepts build flags to control the installation. See 'go help build'.

When checking out or updating a package, get looks for a branch or tag
that matches the locally installed version of Go. The most important
rule is that if the local installation is running version "go1", get
searches for a branch or tag named "go1". If no such version exists it
retrieves the most recent version of the package.

For more about specifying packages, see 'go help packages'.

For more about how 'go get' finds source code to
download, see 'go help importpath'.

See also: go build, go install, go clean.
	`,
}

var getD = cmdGet.Flag.Bool("d", false, "")
var getF = cmdGet.Flag.Bool("f", false, "")
var getT = cmdGet.Flag.Bool("t", false, "")
var getU = cmdGet.Flag.Bool("u", false, "")
var getFix = cmdGet.Flag.Bool("fix", false, "")

func init() {
	addBuildFlags(cmdGet)
	cmdGet.Run = runGet // break init loop
}

func runGet(cmd *Command, args []string) {
	if *getF && !*getU {
		fatalf("go get: cannot use -f flag without -u")
	}

	// Phase 1.  Download/update.
	var stk importStack
	for _, arg := range downloadPaths(args) {
		download(arg, &stk, *getT)
	}
	exitIfErrors()

	// Phase 2. Rescan packages and re-evaluate args list.

	// Code we downloaded and all code that depends on it
	// needs to be evicted from the package cache so that
	// the information will be recomputed.  Instead of keeping
	// track of the reverse dependency information, evict
	// everything.
	for name := range packageCache {
		delete(packageCache, name)
	}

	args = importPaths(args)

	// Phase 3.  Install.
	if *getD {
		// Download only.
		// Check delayed until now so that importPaths
		// has a chance to print errors.
		return
	}

	runInstall(cmd, args)
}

// downloadPaths prepares the list of paths to pass to download.
// It expands ... patterns that can be expanded.  If there is no match
// for a particular pattern, downloadPaths leaves it in the result list,
// in the hope that we can figure out the repository from the
// initial ...-free prefix.
func downloadPaths(args []string) []string {
	args = importPathsNoDotExpansion(args)
	var out []string
	for _, a := range args {
		if strings.Contains(a, "...") {
			var expand []string
			// Use matchPackagesInFS to avoid printing
			// warnings.  They will be printed by the
			// eventual call to importPaths instead.
			if build.IsLocalImport(a) {
				expand = matchPackagesInFS(a)
			} else {
				expand = matchPackages(a)
			}
			if len(expand) > 0 {
				out = append(out, expand...)
				continue
			}
		}
		out = append(out, a)
	}
	return out
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
func download(arg string, stk *importStack, getTestDeps bool) {
	p := loadPackage(arg, stk)
	if p.Error != nil && p.Error.hard {
		errorf("%s", p.Error)
		return
	}

	// There's nothing to do if this is a package in the standard library.
	if p.Standard {
		return
	}

	// Only process each package once.
	// (Unless we're fetching test dependencies for this package,
	// in which case we want to process it again.)
	if downloadCache[arg] && !getTestDeps {
		return
	}
	downloadCache[arg] = true

	pkgs := []*Package{p}
	wildcardOkay := len(*stk) == 0
	isWildcard := false

	// Download if the package is missing, or update if we're using -u.
	if p.Dir == "" || *getU {
		// The actual download.
		stk.push(p.ImportPath)
		err := downloadPackage(p)
		if err != nil {
			errorf("%s", &PackageError{ImportStack: stk.copy(), Err: err.Error()})
			stk.pop()
			return
		}

		args := []string{arg}
		// If the argument has a wildcard in it, re-evaluate the wildcard.
		// We delay this until after reloadPackage so that the old entry
		// for p has been replaced in the package cache.
		if wildcardOkay && strings.Contains(arg, "...") {
			if build.IsLocalImport(arg) {
				args = matchPackagesInFS(arg)
			} else {
				args = matchPackages(arg)
			}
			isWildcard = true
		}

		// Clear all relevant package cache entries before
		// doing any new loads.
		for _, arg := range args {
			p := packageCache[arg]
			if p != nil {
				delete(packageCache, p.Dir)
				delete(packageCache, p.ImportPath)
			}
		}

		pkgs = pkgs[:0]
		for _, arg := range args {
			stk.push(arg)
			p := loadPackage(arg, stk)
			stk.pop()
			if p.Error != nil {
				errorf("%s", p.Error)
				continue
			}
			pkgs = append(pkgs, p)
		}
	}

	// Process package, which might now be multiple packages
	// due to wildcard expansion.
	for _, p := range pkgs {
		if *getFix {
			run(stringList(tool("fix"), relPaths(p.allgofiles)))

			// The imports might have changed, so reload again.
			p = reloadPackage(arg, stk)
			if p.Error != nil {
				errorf("%s", p.Error)
				return
			}
		}

		if isWildcard {
			// Report both the real package and the
			// wildcard in any error message.
			stk.push(p.ImportPath)
		}

		// Process dependencies, now that we know what they are.
		for _, dep := range p.deps {
			// Don't get test dependencies recursively.
			download(dep.ImportPath, stk, false)
		}
		if getTestDeps {
			// Process test dependencies when -t is specified.
			// (Don't get test dependencies for test dependencies.)
			for _, path := range p.TestImports {
				download(path, stk, false)
			}
			for _, path := range p.XTestImports {
				download(path, stk, false)
			}
		}

		if isWildcard {
			stk.pop()
		}
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
		if err != nil {
			return err
		}
		repo = "<local>" // should be unused; make distinctive

		// Double-check where it came from.
		if *getU && vcs.remoteRepo != nil && !*getF {
			dir := filepath.Join(p.build.SrcRoot, rootPath)
			if remote, err := vcs.remoteRepo(vcs, dir); err == nil {
				if rr, err := repoRootForImportPath(p.ImportPath); err == nil {
					repo := rr.repo
					if rr.vcs.resolveRepo != nil {
						resolved, err := rr.vcs.resolveRepo(rr.vcs, dir, repo)
						if err == nil {
							repo = resolved
						}
					}
					if remote != repo {
						return fmt.Errorf("%s is a custom import path for %s, but %s is checked out from %s", rr.root, repo, dir, remote)
					}
				}
			}
		}
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
		// Package not found.  Put in first directory of $GOPATH.
		list := filepath.SplitList(buildContext.GOPATH)
		if len(list) == 0 {
			return fmt.Errorf("cannot download, $GOPATH not set. For more details see: go help gopath")
		}
		// Guard against people setting GOPATH=$GOROOT.
		if list[0] == goroot {
			return fmt.Errorf("cannot download, $GOPATH must not be set to $GOROOT. For more details see: go help gopath")
		}
		p.build.SrcRoot = filepath.Join(list[0], "src")
		p.build.PkgRoot = filepath.Join(list[0], "pkg")
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
		if err = os.MkdirAll(parent, 0777); err != nil {
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

// goTag matches go release tags such as go1 and go1.2.3.
// The numbers involved must be small (at most 4 digits),
// have no unnecessary leading zeros, and the version cannot
// end in .0 - it is go1, not go1.0 or go1.0.0.
var goTag = regexp.MustCompile(
	`^go((0|[1-9][0-9]{0,3})\.)*([1-9][0-9]{0,3})$`,
)

// selectTag returns the closest matching tag for a given version.
// Closest means the latest one that is not after the current release.
// Version "goX" (or "goX.Y" or "goX.Y.Z") matches tags of the same form.
// Version "release.rN" matches tags of the form "go.rN" (N being a floating-point number).
// Version "weekly.YYYY-MM-DD" matches tags like "go.weekly.YYYY-MM-DD".
//
// NOTE(rsc): Eventually we will need to decide on some logic here.
// For now, there is only "go1".  This matches the docs in go help get.
func selectTag(goVersion string, tags []string) (match string) {
	for _, t := range tags {
		if t == "go1" {
			return "go1"
		}
	}
	return ""

	/*
		if goTag.MatchString(goVersion) {
			v := goVersion
			for _, t := range tags {
				if !goTag.MatchString(t) {
					continue
				}
				if cmpGoVersion(match, t) < 0 && cmpGoVersion(t, v) <= 0 {
					match = t
				}
			}
		}

		return match
	*/
}

// cmpGoVersion returns -1, 0, +1 reporting whether
// x < y, x == y, or x > y.
func cmpGoVersion(x, y string) int {
	// Malformed strings compare less than well-formed strings.
	if !goTag.MatchString(x) {
		return -1
	}
	if !goTag.MatchString(y) {
		return +1
	}

	// Compare numbers in sequence.
	xx := strings.Split(x[len("go"):], ".")
	yy := strings.Split(y[len("go"):], ".")

	for i := 0; i < len(xx) && i < len(yy); i++ {
		// The Atoi are guaranteed to succeed
		// because the versions match goTag.
		xi, _ := strconv.Atoi(xx[i])
		yi, _ := strconv.Atoi(yy[i])
		if xi < yi {
			return -1
		} else if xi > yi {
			return +1
		}
	}

	if len(xx) < len(yy) {
		return -1
	}
	if len(xx) > len(yy) {
		return +1
	}
	return 0
}
