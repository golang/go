// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"errors"
	"exec"
	"flag"
	"fmt"
	"go/build"
	"go/token"
	"io/ioutil"
	"os"
	"path/filepath" // use for file system paths
	"regexp"
	"runtime"
	"strings"
)

func usage() {
	fmt.Fprintln(os.Stderr, "usage: goinstall [flags] importpath...")
	fmt.Fprintln(os.Stderr, "       goinstall [flags] -a")
	flag.PrintDefaults()
	os.Exit(2)
}

const logfile = "goinstall.log"

var (
	fset          = token.NewFileSet()
	argv0         = os.Args[0]
	errors_       = false
	parents       = make(map[string]string)
	visit         = make(map[string]status)
	installedPkgs = make(map[string]map[string]bool)
	schemeRe      = regexp.MustCompile(`^[a-z]+://`)

	allpkg            = flag.Bool("a", false, "install all previously installed packages")
	reportToDashboard = flag.Bool("dashboard", true, "report public packages at "+dashboardURL)
	update            = flag.Bool("u", false, "update already-downloaded packages")
	doInstall         = flag.Bool("install", true, "build and install")
	clean             = flag.Bool("clean", false, "clean the package directory before installing")
	nuke              = flag.Bool("nuke", false, "clean the package directory and target before installing")
	useMake           = flag.Bool("make", true, "use make to build and install")
	verbose           = flag.Bool("v", false, "verbose")
)

type status int // status for visited map
const (
	unvisited status = iota
	visiting
	done
)

func logf(format string, args ...interface{}) {
	format = "%s: " + format
	args = append([]interface{}{argv0}, args...)
	fmt.Fprintf(os.Stderr, format, args...)
}

func printf(format string, args ...interface{}) {
	if *verbose {
		logf(format, args...)
	}
}

func errorf(format string, args ...interface{}) {
	errors_ = true
	logf(format, args...)
}

func terrorf(tree *build.Tree, format string, args ...interface{}) {
	if tree != nil && tree.Goroot && os.Getenv("GOPATH") == "" {
		format = strings.TrimRight(format, "\n") + " ($GOPATH not set)\n"
	}
	errorf(format, args...)
}

func main() {
	flag.Usage = usage
	flag.Parse()
	if runtime.GOROOT() == "" {
		fmt.Fprintf(os.Stderr, "%s: no $GOROOT\n", argv0)
		os.Exit(1)
	}
	readPackageList()

	// special case - "unsafe" is already installed
	visit["unsafe"] = done

	args := flag.Args()
	if *allpkg {
		if len(args) != 0 {
			usage() // -a and package list both provided
		}
		// install all packages that were ever installed
		n := 0
		for _, pkgs := range installedPkgs {
			for pkg := range pkgs {
				args = append(args, pkg)
				n++
			}
		}
		if n == 0 {
			logf("no installed packages\n")
			os.Exit(1)
		}
	}
	if len(args) == 0 {
		usage()
	}
	for _, path := range args {
		if s := schemeRe.FindString(path); s != "" {
			errorf("%q used in import path, try %q\n", s, path[len(s):])
			continue
		}

		install(path, "")
	}
	if errors_ {
		os.Exit(1)
	}
}

// printDeps prints the dependency path that leads to pkg.
func printDeps(pkg string) {
	if pkg == "" {
		return
	}
	if visit[pkg] != done {
		printDeps(parents[pkg])
	}
	fmt.Fprintf(os.Stderr, "\t%s ->\n", pkg)
}

// readPackageList reads the list of installed packages from the
// goinstall.log files in GOROOT and the GOPATHs and initalizes
// the installedPkgs variable.
func readPackageList() {
	for _, t := range build.Path {
		installedPkgs[t.Path] = make(map[string]bool)
		name := filepath.Join(t.Path, logfile)
		pkglistdata, err := ioutil.ReadFile(name)
		if err != nil {
			printf("%s\n", err)
			continue
		}
		pkglist := strings.Fields(string(pkglistdata))
		for _, pkg := range pkglist {
			installedPkgs[t.Path][pkg] = true
		}
	}
}

// logPackage logs the named package as installed in the goinstall.log file
// in the given tree if the package is not already in that file.
func logPackage(pkg string, tree *build.Tree) (logged bool) {
	if installedPkgs[tree.Path][pkg] {
		return false
	}
	name := filepath.Join(tree.Path, logfile)
	fout, err := os.OpenFile(name, os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0666)
	if err != nil {
		terrorf(tree, "package log: %s\n", err)
		return false
	}
	fmt.Fprintf(fout, "%s\n", pkg)
	fout.Close()
	return true
}

// install installs the package named by path, which is needed by parent.
func install(pkg, parent string) {
	// Make sure we're not already trying to install pkg.
	switch visit[pkg] {
	case done:
		return
	case visiting:
		fmt.Fprintf(os.Stderr, "%s: package dependency cycle\n", argv0)
		printDeps(parent)
		fmt.Fprintf(os.Stderr, "\t%s\n", pkg)
		os.Exit(2)
	}
	parents[pkg] = parent
	visit[pkg] = visiting
	defer func() {
		visit[pkg] = done
	}()

	// Don't allow trailing '/'
	if strings.HasSuffix(pkg, "/") {
		errorf("%s should not have trailing '/'\n", pkg)
		return
	}

	// Check whether package is local or remote.
	// If remote, download or update it.
	tree, pkg, err := build.FindTree(pkg)
	// Don't build the standard library.
	if err == nil && tree.Goroot && isStandardPath(pkg) {
		if parent == "" {
			errorf("%s: can not goinstall the standard library\n", pkg)
		} else {
			printf("%s: skipping standard library\n", pkg)
		}
		return
	}
	// Download remote packages if not found or forced with -u flag.
	remote, public := isRemote(pkg), false
	if remote {
		if err == build.ErrNotFound || (err == nil && *update) {
			// Download remote package.
			printf("%s: download\n", pkg)
			public, err = download(pkg, tree.SrcDir())
		} else {
			// Test if this is a public repository
			// (for reporting to dashboard).
			m, _ := findPublicRepo(pkg)
			public = m != nil
		}
	}
	if err != nil {
		terrorf(tree, "%s: %v\n", pkg, err)
		return
	}
	dir := filepath.Join(tree.SrcDir(), filepath.FromSlash(pkg))

	// Install prerequisites.
	dirInfo, err := build.ScanDir(dir)
	if err != nil {
		terrorf(tree, "%s: %v\n", pkg, err)
		return
	}
	// We reserve package main to identify commands.
	if parent != "" && dirInfo.Package == "main" {
		terrorf(tree, "%s: found only package main in %s; cannot import", pkg, dir)
		return
	}
	for _, p := range dirInfo.Imports {
		if p != "C" {
			install(p, pkg)
		}
	}
	if errors_ {
		return
	}

	// Install this package.
	if *useMake {
		err := domake(dir, pkg, tree, dirInfo.IsCommand())
		if err != nil {
			terrorf(tree, "%s: install: %v\n", pkg, err)
			return
		}
	} else {
		script, err := build.Build(tree, pkg, dirInfo)
		if err != nil {
			terrorf(tree, "%s: install: %v\n", pkg, err)
			return
		}
		if *nuke {
			printf("%s: nuke\n", pkg)
			script.Nuke()
		} else if *clean {
			printf("%s: clean\n", pkg)
			script.Clean()
		}
		if *doInstall {
			if script.Stale() {
				printf("%s: install\n", pkg)
				if err := script.Run(); err != nil {
					terrorf(tree, "%s: install: %v\n", pkg, err)
					return
				}
			} else {
				printf("%s: up-to-date\n", pkg)
			}
		}
	}

	if remote {
		// mark package as installed in goinstall.log
		logged := logPackage(pkg, tree)

		// report installation to the dashboard if this is the first
		// install from a public repository.
		if logged && public {
			maybeReportToDashboard(pkg)
		}
	}
}

// Is this a standard package path?  strings container/list etc.
// Assume that if the first element has a dot, it's a domain name
// and is not the standard package path.
func isStandardPath(s string) bool {
	dot := strings.Index(s, ".")
	slash := strings.Index(s, "/")
	return dot < 0 || 0 < slash && slash < dot
}

// run runs the command cmd in directory dir with standard input stdin.
// If the command fails, run prints the command and output on standard error
// in addition to returning a non-nil error.
func run(dir string, stdin []byte, cmd ...string) error {
	return genRun(dir, stdin, cmd, false)
}

// quietRun is like run but prints nothing on failure unless -v is used.
func quietRun(dir string, stdin []byte, cmd ...string) error {
	return genRun(dir, stdin, cmd, true)
}

// genRun implements run and quietRun.
func genRun(dir string, stdin []byte, arg []string, quiet bool) error {
	cmd := exec.Command(arg[0], arg[1:]...)
	cmd.Stdin = bytes.NewBuffer(stdin)
	cmd.Dir = dir
	printf("%s: %s %s\n", dir, cmd.Path, strings.Join(arg[1:], " "))
	out, err := cmd.CombinedOutput()
	if err != nil {
		if !quiet || *verbose {
			if dir != "" {
				dir = "cd " + dir + "; "
			}
			fmt.Fprintf(os.Stderr, "%s: === %s%s\n", cmd.Path, dir, strings.Join(cmd.Args, " "))
			os.Stderr.Write(out)
			fmt.Fprintf(os.Stderr, "--- %s\n", err)
		}
		return errors.New("running " + arg[0] + ": " + err.Error())
	}
	return nil
}
