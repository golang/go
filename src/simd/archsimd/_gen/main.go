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

const defaultLocalISA = "ISA_A64_xml_A_profile-2026-03_96/ISA_A64_xml_A_profile_2026-03_96-2026-03_rel" // ISA_A64_xml_A_profile-2025-06"
const defaultXedPath = "$XEDPATH" + string(filepath.ListSeparator) + "./simdgen/xeddata" + string(filepath.ListSeparator) + "$HOME/xed/obj/dgen"
const defaultArm64Path = "$ARM64_ISA_PATH" + string(filepath.ListSeparator) + "./simdgen/armdata" + string(filepath.ListSeparator) + "$HOME/Downloads/" + defaultLocalISA

var (
	flagTmplgen = flag.Bool("tmplgen", true, "run tmplgen generator")
	flagSimdgen = flag.Bool("simdgen", true, "run simdgen generator")
	flagWasmgen = flag.Bool("wasmgen", true, "run wasmgen generator")
	flagMidway  = flag.Bool("midway", true, "run midway generator")

	flagN         = flag.Bool("n", false, "dry run")
	flagXedPath   = flag.String("xedPath", defaultXedPath, "load XED datafile from `path`, which must be the XED obj/dgen directory")
	flagArm64Path = flag.String("arm64Path", defaultArm64Path, "load ARM64 ISA XML definitions from `path`")

	flagGoRoot = flag.String("goroot", "", "destination go dev tree for generated files")
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

	if *flagArm64Path == defaultArm64Path {
		// In general we want the shell to do variable expansion, but for the
		// default value we don't get that, so do it ourselves.
		*flagArm64Path = os.ExpandEnv(defaultArm64Path)
	}

	var err error
	goRoot, err = resolveGOROOT()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	if *flagSimdgen {
		fmt.Fprintln(os.Stderr, "# This may take a few minutes...")
	}

	if *flagTmplgen {
		doTmplgen()
	}

	if *flagWasmgen || *flagSimdgen {
		ssaGenPath := prettyPath(".", filepath.Join(goRoot, "src", "cmd", "compile", "internal", "ssa", "_gen"))

		// If there is garbage in ssa/_gen/simdgenericOps.go, it can affect the merge in simdgen/wasmgen.
		removeSimdGenericOps(ssaGenPath)

		if *flagWasmgen {
			doWasmgen()
		}
		if *flagSimdgen {
			doSimdgen()
		}
		ssaGen(ssaGenPath)
	}

	if *flagMidway {
		doMidway()
	}
}

func removeSimdGenericOps(ssaGenPath string) {
	ssaSimdGenericOps := filepath.Join(ssaGenPath, "simdgenericOps.go")
	if _, err := os.Stat(ssaSimdGenericOps); err == nil {
		if err = os.Remove(ssaSimdGenericOps); err != nil {
			fmt.Fprintf(os.Stderr, "Failed to delete %s before regenerating it, %v\n", ssaSimdGenericOps, err)
			os.Exit(1)
		}
	}
}

func ssaGen(ssaGenPath string) {
	// simdgen produces SSA rule files, so update the SSA files
	goRun("-C", ssaGenPath, ".")

	fmt.Fprintln(os.Stderr, "# Compiler changed. Consider running \"go install cmd/compile\"")
}

func doTmplgen() {
	goRun("-C", "tmplgen", ".")
}

func doWasmgen() {
	goRun("-C", "wasmgen", ".")
}

func doMidway() {
	goRun("-C", "midway", ".")
}

func doSimdgen() {
	xedPath, err := resolveXEDPath(*flagXedPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	armPath, err := resolveARMPath(*flagArm64Path)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	goRun("-C", "simdgen", ".", "-o", "godefs", "-goroot", goRoot, "-arch", "arm64", "-arm64Path", prettyPath("./simdgen", armPath), "go_arm64.yaml", "types.yaml", "categories.yaml")

	// Regenerate the XED-derived SIMD files
	goRun("-C", "simdgen", ".", "-o", "godefs", "-goroot", goRoot, "-arch", "amd64", "-xedPath", prettyPath("./simdgen", xedPath), "go_amd64.yaml", "types.yaml", "categories.yaml")
}

// simdgen -o godefs -goroot goRoot -arm64Path $ARM64_ISA_PATH arm64/go.yaml arm64/categories.yaml types.yaml

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

func resolveARMPath(pathList string) (armPath string, err error) {
	for _, path := range filepath.SplitList(pathList) {
		if path == "" {
			// Probably an unknown shell variable. Ignore.
			continue
		}
		if _, err := os.Stat(filepath.Join(path, "abs_advsimd.xml")); err == nil {
			return filepath.Abs(path)
		}
	}
	return "", fmt.Errorf("set $ARM64_ISA_PATH or -armPath to the ARM64 ISA specification directory")
}

func resolveGOROOT() (goRoot string, err error) {
	goRoot = *flagGoRoot
	if goRoot != "" {
		return
	}
	// Using the current compiler's goroot depends on a working dev compiler,
	// which is not guaranteed.  Instead, assume
	cwd, err := os.Getwd()
	if err != nil {
		return "", fmt.Errorf("Getwd error: %s", err)
	}
	goRoot, err = filepath.Abs(filepath.Join(cwd, "..", "..", "..", ".."))
	if err != nil {
		return "", fmt.Errorf("Abs path error: %s", err)
	}
	_, err = os.Stat(filepath.Join(goRoot, "src", "simd", "archsimd", "_gen"))
	if err != nil {
		return "", fmt.Errorf("-goroot not specified and not run in src/simd/archsimd/_gen")
	}

	return
}

func goRun(args ...string) {
	exe := "go" // Use go on the path, not GOROOT.  GOROOT could be broken
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
		os.Exit(1)
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
