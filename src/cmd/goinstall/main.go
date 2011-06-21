// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"exec"
	"flag"
	"fmt"
	"go/build"
	"go/token"
	"io/ioutil"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

func usage() {
	fmt.Fprint(os.Stderr, "usage: goinstall importpath...\n")
	fmt.Fprintf(os.Stderr, "\tgoinstall -a\n")
	flag.PrintDefaults()
	os.Exit(2)
}

var (
	fset          = token.NewFileSet()
	argv0         = os.Args[0]
	errors        = false
	parents       = make(map[string]string)
	visit         = make(map[string]status)
	logfile       = filepath.Join(runtime.GOROOT(), "goinstall.log")
	installedPkgs = make(map[string]bool)

	allpkg            = flag.Bool("a", false, "install all previously installed packages")
	reportToDashboard = flag.Bool("dashboard", true, "report public packages at "+dashboardURL)
	logPkgs           = flag.Bool("log", true, "log installed packages to $GOROOT/goinstall.log for use by -a")
	update            = flag.Bool("u", false, "update already-downloaded packages")
	doInstall         = flag.Bool("install", true, "build and install")
	clean             = flag.Bool("clean", false, "clean the package directory before installing")
	nuke              = flag.Bool("nuke", false, "clean the package directory and target before installing")
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
	errors = true
	logf(format, args...)
}

func main() {
	flag.Usage = usage
	flag.Parse()
	if runtime.GOROOT() == "" {
		fmt.Fprintf(os.Stderr, "%s: no $GOROOT\n", argv0)
		os.Exit(1)
	}

	// special case - "unsafe" is already installed
	visit["unsafe"] = done

	args := flag.Args()
	if *allpkg || *logPkgs {
		readPackageList()
	}
	if *allpkg {
		if len(args) != 0 {
			usage() // -a and package list both provided
		}
		// install all packages that were ever installed
		if len(installedPkgs) == 0 {
			fmt.Fprintf(os.Stderr, "%s: no installed packages\n", argv0)
			os.Exit(1)
		}
		args = make([]string, len(installedPkgs), len(installedPkgs))
		i := 0
		for pkg := range installedPkgs {
			args[i] = pkg
			i++
		}
	}
	if len(args) == 0 {
		usage()
	}
	for _, path := range args {
		if strings.HasPrefix(path, "http://") {
			errorf("'http://' used in remote path, try '%s'\n", path[7:])
			continue
		}

		install(path, "")
	}
	if errors {
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

// readPackageList reads the list of installed packages from goinstall.log
func readPackageList() {
	pkglistdata, _ := ioutil.ReadFile(logfile)
	pkglist := strings.Fields(string(pkglistdata))
	for _, pkg := range pkglist {
		installedPkgs[pkg] = true
	}
}

// logPackage logs the named package as installed in goinstall.log, if the package is not found in there
func logPackage(pkg string) {
	if installedPkgs[pkg] {
		return
	}
	fout, err := os.OpenFile(logfile, os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0666)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s: %s\n", argv0, err)
		return
	}
	fmt.Fprintf(fout, "%s\n", pkg)
	fout.Close()
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
	remote := isRemote(pkg)
	if remote && (err == build.ErrNotFound || (err == nil && *update)) {
		printf("%s: download\n", pkg)
		err = download(pkg, tree.SrcDir())
	}
	if err != nil {
		errorf("%s: %v\n", pkg, err)
		return
	}
	dir := filepath.Join(tree.SrcDir(), pkg)

	// Install prerequisites.
	dirInfo, err := build.ScanDir(dir, parent == "")
	if err != nil {
		errorf("%s: %v\n", pkg, err)
		return
	}
	if len(dirInfo.GoFiles)+len(dirInfo.CgoFiles) == 0 {
		errorf("%s: package has no files\n", pkg)
		return
	}
	for _, p := range dirInfo.Imports {
		if p != "C" {
			install(p, pkg)
		}
	}
	if errors {
		return
	}

	// Install this package.
	script, err := build.Build(tree, pkg, dirInfo)
	if err != nil {
		errorf("%s: install: %v\n", pkg, err)
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
				errorf("%s: install: %v\n", pkg, err)
				return
			}
		} else {
			printf("%s: up-to-date\n", pkg)
		}
	}
	if remote {
		// mark package as installed in $GOROOT/goinstall.log
		logPackage(pkg)
	}
	return
}


// Is this a standard package path?  strings container/vector etc.
// Assume that if the first element has a dot, it's a domain name
// and is not the standard package path.
func isStandardPath(s string) bool {
	dot := strings.Index(s, ".")
	slash := strings.Index(s, "/")
	return dot < 0 || 0 < slash && slash < dot
}

// run runs the command cmd in directory dir with standard input stdin.
// If the command fails, run prints the command and output on standard error
// in addition to returning a non-nil os.Error.
func run(dir string, stdin []byte, cmd ...string) os.Error {
	return genRun(dir, stdin, cmd, false)
}

// quietRun is like run but prints nothing on failure unless -v is used.
func quietRun(dir string, stdin []byte, cmd ...string) os.Error {
	return genRun(dir, stdin, cmd, true)
}

// genRun implements run and quietRun.
func genRun(dir string, stdin []byte, arg []string, quiet bool) os.Error {
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
		return os.ErrorString("running " + arg[0] + ": " + err.String())
	}
	return nil
}
