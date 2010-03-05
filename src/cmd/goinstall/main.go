// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Experimental Go package installer; see doc.go.

package main

import (
	"bytes"
	"exec"
	"flag"
	"fmt"
	"io"
	"os"
	"path"
	"strings"
)

func usage() {
	fmt.Fprint(os.Stderr, "usage: goinstall importpath...\n")
	flag.PrintDefaults()
	os.Exit(2)
}

var (
	argv0   = os.Args[0]
	errors  = false
	gobin   = os.Getenv("GOBIN")
	parents = make(map[string]string)
	root    = os.Getenv("GOROOT")
	visit   = make(map[string]status)

	reportToDashboard = flag.Bool("dashboard", true, "report public packages at "+dashboardURL)
	update            = flag.Bool("u", false, "update already-downloaded packages")
	verbose           = flag.Bool("v", false, "verbose")
)

type status int // status for visited map
const (
	unvisited status = iota
	visiting
	done
)

func main() {
	flag.Usage = usage
	flag.Parse()
	if root == "" {
		fmt.Fprintf(os.Stderr, "%s: no $GOROOT\n", argv0)
		os.Exit(1)
	}
	root += "/src/pkg/"
	if gobin == "" {
		gobin = os.Getenv("HOME") + "/bin"
	}

	// special case - "unsafe" is already installed
	visit["unsafe"] = done

	// install command line arguments
	args := flag.Args()
	if len(args) == 0 {
		usage()
	}
	for _, path := range args {
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

// install installs the package named by path, which is needed by parent.
func install(pkg, parent string) {
	// Make sure we're not already trying to install pkg.
	switch v, _ := visit[pkg]; v {
	case done:
		return
	case visiting:
		fmt.Fprintf(os.Stderr, "%s: package dependency cycle\n", argv0)
		printDeps(parent)
		fmt.Fprintf(os.Stderr, "\t%s\n", pkg)
		os.Exit(2)
	}
	visit[pkg] = visiting
	parents[pkg] = parent
	if *verbose {
		fmt.Println(pkg)
	}

	// Check whether package is local or remote.
	// If remote, download or update it.
	var dir string
	local := false
	if isLocalPath(pkg) {
		dir = pkg
		local = true
	} else if isStandardPath(pkg) {
		dir = path.Join(root, pkg)
		local = true
	} else {
		var err os.Error
		dir, err = download(pkg)
		if err != nil {
			fmt.Fprintf(os.Stderr, "%s: %s: %s\n", argv0, pkg, err)
			errors = true
			visit[pkg] = done
			return
		}
	}

	// Install prerequisites.
	files, m, err := goFiles(dir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s: %s: %s\n", argv0, pkg, err)
		errors = true
		visit[pkg] = done
		return
	}
	if len(files) == 0 {
		fmt.Fprintf(os.Stderr, "%s: %s: package has no files\n", argv0, pkg)
		errors = true
		visit[pkg] = done
		return
	}
	for p := range m {
		install(p, pkg)
	}

	// Install this package.
	if !errors {
		if err := domake(dir, pkg, local); err != nil {
			fmt.Fprintf(os.Stderr, "%s: installing %s: %s\n", argv0, pkg, err)
			errors = true
		}
	}

	visit[pkg] = done
}

// Is this a local path?  /foo ./foo ../foo . ..
func isLocalPath(s string) bool {
	return strings.HasPrefix(s, "/") || strings.HasPrefix(s, "./") || strings.HasPrefix(s, "../") || s == "." || s == ".."
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

// genRun implements run and tryRun.
func genRun(dir string, stdin []byte, cmd []string, quiet bool) os.Error {
	bin, err := exec.LookPath(cmd[0])
	if err != nil {
		return err
	}
	p, err := exec.Run(bin, cmd, os.Environ(), dir, exec.Pipe, exec.Pipe, exec.MergeWithStdout)
	if *verbose {
		fmt.Fprintf(os.Stderr, "%s: %s; %s %s\n", argv0, dir, bin, strings.Join(cmd[1:], " "))
	}
	if err != nil {
		return err
	}
	go func() {
		p.Stdin.Write(stdin)
		p.Stdin.Close()
	}()
	var buf bytes.Buffer
	io.Copy(&buf, p.Stdout)
	io.Copy(&buf, p.Stdout)
	w, err := p.Wait(0)
	p.Close()
	if !w.Exited() || w.ExitStatus() != 0 {
		if !quiet || *verbose {
			if dir != "" {
				dir = "cd " + dir + "; "
			}
			fmt.Fprintf(os.Stderr, "%s: === %s%s\n", argv0, dir, strings.Join(cmd, " "))
			os.Stderr.Write(buf.Bytes())
			fmt.Fprintf(os.Stderr, "--- %s\n", w)
		}
		return os.ErrorString("running " + cmd[0] + ": " + w.String())
	}
	return nil
}
