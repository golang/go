// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"errors"
	"flag"
	"fmt"
	"go/build"
	"go/token"
	"io/ioutil"
	"os"
	"os/exec"
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
	parents       = make(map[string]string)
	visit         = make(map[string]status)
	installedPkgs = make(map[string]map[string]bool)
	schemeRe      = regexp.MustCompile(`^[a-z]+://`)

	allpkg            = flag.Bool("a", false, "install all previously installed packages")
	reportToDashboard = flag.Bool("dashboard", true, "report public packages at "+dashboardURL)
	update            = flag.Bool("u", false, "update already-downloaded packages")
	doGofix           = flag.Bool("fix", false, "gofix each package before building it")
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

type PackageError struct {
	pkg string
	err error
}

func (e *PackageError) Error() string {
	return fmt.Sprintf("%s: %v", e.pkg, e.err)
}

type DownloadError struct {
	pkg    string
	goroot bool
	err    error
}

func (e *DownloadError) Error() string {
	s := fmt.Sprintf("%s: download failed: %v", e.pkg, e.err)
	if e.goroot && os.Getenv("GOPATH") == "" {
		s += " ($GOPATH is not set)"
	}
	return s
}

type DependencyError PackageError

func (e *DependencyError) Error() string {
	return fmt.Sprintf("%s: depends on failing packages:\n\t%v", e.pkg, e.err)
}

type BuildError PackageError

func (e *BuildError) Error() string {
	return fmt.Sprintf("%s: build failed: %v", e.pkg, e.err)
}

type RunError struct {
	cmd, dir string
	out      []byte
	err      error
}

func (e *RunError) Error() string {
	return fmt.Sprintf("%v\ncd %q && %q\n%s", e.err, e.dir, e.cmd, e.out)
}

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
	errs := false
	for _, path := range args {
		if err := install(path, ""); err != nil {
			errs = true
			fmt.Fprintln(os.Stderr, err)
		}
	}
	if errs {
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
// goinstall.log files in GOROOT and the GOPATHs and initializes
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
		printf("package log: %s\n", err)
		return false
	}
	fmt.Fprintf(fout, "%s\n", pkg)
	fout.Close()
	return true
}

// install installs the package named by path, which is needed by parent.
func install(pkg, parent string) error {
	// Basic validation of import path string.
	if s := schemeRe.FindString(pkg); s != "" {
		return fmt.Errorf("%q used in import path, try %q\n", s, pkg[len(s):])
	}
	if strings.HasSuffix(pkg, "/") {
		return fmt.Errorf("%q should not have trailing '/'\n", pkg)
	}

	// Make sure we're not already trying to install pkg.
	switch visit[pkg] {
	case done:
		return nil
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

	// Check whether package is local or remote.
	// If remote, download or update it.
	tree, pkg, err := build.FindTree(pkg)
	// Don't build the standard library.
	if err == nil && tree.Goroot && isStandardPath(pkg) {
		if parent == "" {
			return &PackageError{pkg, errors.New("cannot goinstall the standard library")}
		}
		return nil
	}

	// Download remote packages if not found or forced with -u flag.
	remote, public := isRemote(pkg), false
	if remote {
		if err == build.ErrNotFound || (err == nil && *update) {
			// Download remote package.
			printf("%s: download\n", pkg)
			public, err = download(pkg, tree.SrcDir())
			if err != nil {
				// only suggest -fix if the bad import was not on the command line
				if e, ok := err.(*errOldGoogleRepo); ok && parent != "" {
					err = fmt.Errorf("%v\nRun goinstall with -fix to gofix the code.", e)
				}
				return &DownloadError{pkg, tree.Goroot, err}
			}
		} else {
			// Test if this is a public repository
			// (for reporting to dashboard).
			repo, e := findPublicRepo(pkg)
			public = repo != nil
			err = e
		}
	}
	if err != nil {
		return &PackageError{pkg, err}
	}

	// Install the package and its dependencies.
	if err := installPackage(pkg, parent, tree, false); err != nil {
		return err
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

	return nil
}

// installPackage installs the specified package and its dependencies.
func installPackage(pkg, parent string, tree *build.Tree, retry bool) (installErr error) {
	printf("%s: install\n", pkg)

	// Read package information.
	dir := filepath.Join(tree.SrcDir(), filepath.FromSlash(pkg))
	dirInfo, err := build.ScanDir(dir)
	if err != nil {
		return &PackageError{pkg, err}
	}

	// We reserve package main to identify commands.
	if parent != "" && dirInfo.Package == "main" {
		return &PackageError{pkg, fmt.Errorf("found only package main in %s; cannot import", dir)}
	}

	// Run gofix if we fail to build and -fix is set.
	defer func() {
		if retry || installErr == nil || !*doGofix {
			return
		}
		if e, ok := (installErr).(*DependencyError); ok {
			// If this package failed to build due to a
			// DependencyError, only attempt to gofix it if its
			// dependency failed for some reason other than a
			// DependencyError or BuildError.
			// (If a dep or one of its deps doesn't build there's
			// no way that gofixing this package can help.)
			switch e.err.(type) {
			case *DependencyError:
				return
			case *BuildError:
				return
			}
		}
		gofix(pkg, dir, dirInfo)
		installErr = installPackage(pkg, parent, tree, true) // retry
	}()

	// Install prerequisites.
	for _, p := range dirInfo.Imports {
		if p == "C" {
			continue
		}
		if err := install(p, pkg); err != nil {
			return &DependencyError{pkg, err}
		}
	}

	// Install this package.
	if *useMake {
		err := domake(dir, pkg, tree, dirInfo.IsCommand())
		if err != nil {
			return &BuildError{pkg, err}
		}
		return nil
	}
	script, err := build.Build(tree, pkg, dirInfo)
	if err != nil {
		return &BuildError{pkg, err}
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
				return &BuildError{pkg, err}
			}
		} else {
			printf("%s: up-to-date\n", pkg)
		}
	}

	return nil
}

// gofix runs gofix against the GoFiles and CgoFiles of dirInfo in dir.
func gofix(pkg, dir string, dirInfo *build.DirInfo) {
	printf("%s: gofix\n", pkg)
	files := append([]string{}, dirInfo.GoFiles...)
	files = append(files, dirInfo.CgoFiles...)
	for i, file := range files {
		files[i] = filepath.Join(dir, file)
	}
	cmd := exec.Command("gofix", files...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		logf("%s: gofix: %v", pkg, err)
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
// If verbose is set and the command fails it prints the output to stderr.
func run(dir string, stdin []byte, arg ...string) error {
	cmd := exec.Command(arg[0], arg[1:]...)
	cmd.Stdin = bytes.NewBuffer(stdin)
	cmd.Dir = dir
	printf("cd %s && %s %s\n", dir, cmd.Path, strings.Join(arg[1:], " "))
	if out, err := cmd.CombinedOutput(); err != nil {
		if *verbose {
			fmt.Fprintf(os.Stderr, "%v\n%s\n", err, out)
		}
		return &RunError{strings.Join(arg, " "), dir, out, err}
	}
	return nil
}

// isRemote returns true if the first part of the package name looks like a
// hostname - i.e. contains at least one '.' and the last part is at least 2
// characters.
func isRemote(pkg string) bool {
	parts := strings.SplitN(pkg, "/", 2)
	if len(parts) != 2 {
		return false
	}
	parts = strings.Split(parts[0], ".")
	if len(parts) < 2 || len(parts[len(parts)-1]) < 2 {
		return false
	}
	return true
}
