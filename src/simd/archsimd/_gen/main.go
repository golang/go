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

	"simd/archsimd/_gen/gentools"
	"simd/archsimd/_gen/sgutil"
)

var (
	flagTmplgen = flag.Bool("tmplgen", true, "run tmplgen generator")
	flagSimdgen = flag.Bool("simdgen", true, "run simdgen generator")
	flagWasmgen = flag.Bool("wasmgen", true, "run wasmgen generator")
	flagMidway  = flag.Bool("midway", true, "run midway generator")

	flagN         = flag.Bool("n", false, "dry run")
	flagXedPath   = sgutil.FlagXEDPath(".")
	flagArm64Path = sgutil.FlagARM64Path(".")

	genFlags = gentools.RegisterFlags(nil)
)

func main() {
	flag.Parse()
	if flag.NArg() > 0 {
		flag.Usage()
		os.Exit(1)
	}

	if genFlags.GOROOT == "" {
		fmt.Fprintln(os.Stderr, "failed to find Go dev tree root from current directory")
		os.Exit(1)
	}

	// If we need data paths, resolve them before we start running any tools so
	// we can report errors immediately.
	var xedPath, armPath string
	if *flagSimdgen {
		var err error
		resolveError := false
		xedPath, err = sgutil.ResolveXEDPath(flagXedPath)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			resolveError = true
		}
		armPath, err = sgutil.ResolveARM64Path(flagArm64Path)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			resolveError = true
		}
		if resolveError {
			os.Exit(1)
		}
	}

	if *flagSimdgen {
		fmt.Fprintln(os.Stderr, "# This may take a few minutes...")
	}

	var files gentools.Files
	defer files.FlushOrExit()

	if *flagTmplgen {
		doGen("tmplgen", &files)
	}

	if *flagWasmgen || *flagSimdgen {
		ssaGenPath := prettyPath(".", genFlags.OutputPath("cmd/compile/internal/ssa/_gen"))

		// If there is garbage in ssa/_gen/simdgenericOps.go, it can affect the merge in simdgen/wasmgen.
		if genFlags.Write {
			removeSimdGenericOps(ssaGenPath)
		}

		if *flagWasmgen {
			doGen("wasmgen", &files)
		}
		if *flagSimdgen {
			doSimdgen(xedPath, armPath, &files)
		}

		// ssaGen doesn't use gentools, so we can only run if it we're writing
		// to the Go source tree, and we have to flush any file changes before
		// we do.
		if genFlags.WritingToInput() {
			files.FlushOrExit()
			ssaGen(ssaGenPath)
		} else {
			fmt.Fprintf(os.Stderr, "# skipping %s gen because we're not writing to -goroot\n", ssaGenPath)
		}
	}

	if *flagMidway {
		doGen("midway", &files)
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

func doGen(tool string, files *gentools.Files) {
	flags := append([]string{"-C", tool, "."}, files.ExecFlags()...)
	goRun(flags...)
}

func doSimdgen(xedPath, armPath string, files *gentools.Files) {
	armArgs := append([]string{"-C", "simdgen", ".", "-o", "godefs", "-arch", "arm64", "-arm64Path", prettyPath("./simdgen", armPath)}, files.ExecFlags()...)
	armArgs = append(armArgs, "go_arm64.yaml", "types.yaml", "categories.yaml")
	goRun(armArgs...)

	// Regenerate the XED-derived SIMD files
	amdArgs := append([]string{"-C", "simdgen", ".", "-o", "godefs", "-arch", "amd64", "-xedPath", prettyPath("./simdgen", xedPath)}, files.ExecFlags()...)
	amdArgs = append(amdArgs, "go_amd64.yaml", "types.yaml", "categories.yaml")
	goRun(amdArgs...)
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
