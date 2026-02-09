// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Run all SIMD-related code generators.
package main

import (
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

const defaultXedPath = "$XEDPATH" + string(filepath.ListSeparator) + "./simdgen/xeddata" + string(filepath.ListSeparator) + "$HOME/xed/obj/dgen"

var (
	flagTmplgen = flag.Bool("tmplgen", true, "run tmplgen generator")
	flagSimdgen = flag.Bool("simdgen", true, "run simdgen generator")

	flagN       = flag.Bool("n", false, "dry run")
	flagXedPath = flag.String("xedPath", defaultXedPath, "load XED datafile from `path`, which must be the XED obj/dgen directory")
)

var goRoot string

func main() {
	flag.Parse()
	if flag.NArg() > 0 {
		flag.Usage()
		os.Exit(1)
	}

	if *flagXedPath == defaultXedPath {
		// In general we want the shell to do variable expansion, but for the
		// default value we don't get that, so do it ourselves.
		*flagXedPath = os.ExpandEnv(defaultXedPath)
	}

	var err error
	goRoot, err = resolveGOROOT()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	if *flagTmplgen {
		doTmplgen()
	}
	if *flagSimdgen {
		doSimdgen()
	}
}

func doTmplgen() {
	goRun("-C", "tmplgen", ".")
}

func doSimdgen() {
	xedPath, err := resolveXEDPath(*flagXedPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	// Regenerate the XED-derived SIMD files
	goRun("-C", "simdgen", ".", "-o", "godefs", "-goroot", goRoot, "-xedPath", prettyPath("./simdgen", xedPath), "go.yaml", "types.yaml", "categories.yaml")

	// simdgen produces SSA rule files, so update the SSA files
	goRun("-C", prettyPath(".", filepath.Join(goRoot, "src", "cmd", "compile", "internal", "ssa", "_gen")), ".")
}

func resolveXEDPath(pathList string) (xedPath string, err error) {
	for _, path := range filepath.SplitList(pathList) {
		if path == "" {
			// Probably an unknown shell variable. Ignore.
			continue
		}
		if _, err := os.Stat(filepath.Join(path, "all-dec-instructions.txt")); err == nil {
			return filepath.Abs(path)
		}
	}
	return "", fmt.Errorf("set $XEDPATH or -xedPath to the XED obj/dgen directory")
}

func resolveGOROOT() (goRoot string, err error) {
	cmd := exec.Command("go", "env", "GOROOT")
	cmd.Stderr = os.Stderr
	out, err := cmd.Output()
	if err != nil {
		return "", fmt.Errorf("%s: %s", cmd, err)
	}
	goRoot = strings.TrimSuffix(string(out), "\n")
	return goRoot, nil
}

func goRun(args ...string) {
	exe := filepath.Join(goRoot, "bin", "go")
	cmd := exec.Command(exe, append([]string{"run"}, args...)...)
	run(cmd)
}

func run(cmd *exec.Cmd) {
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	fmt.Fprintf(os.Stderr, "%s\n", cmdString(cmd))
	if *flagN {
		return
	}
	if err := cmd.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "%s failed: %s\n", cmd, err)
	}
}

func prettyPath(base, path string) string {
	base, err := filepath.Abs(base)
	if err != nil {
		return path
	}
	p, err := filepath.Rel(base, path)
	if err != nil {
		return path
	}
	return p
}

func cmdString(cmd *exec.Cmd) string {
	// TODO: Shell quoting?
	// TODO: Environment.

	var buf strings.Builder

	cmdPath, err := exec.LookPath(filepath.Base(cmd.Path))
	if err == nil && cmdPath == cmd.Path {
		cmdPath = filepath.Base(cmdPath)
	} else {
		cmdPath = prettyPath(".", cmd.Path)
	}
	buf.WriteString(cmdPath)

	for _, arg := range cmd.Args[1:] {
		buf.WriteByte(' ')
		buf.WriteString(arg)
	}

	return buf.String()
}
