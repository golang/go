// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Run all SIMD-related code generators.
package main

import (
	"flag"
	"fmt"
	"maps"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"

	"simd/archsimd/_gen/gentools"
	"simd/archsimd/_gen/sgutil"
)

var (
	flagTools = flagVar("tools", ToolSet{"tmplgen": true, "simdgen": true, "wasmgen": true, "midway": true}, "comma-separated list of tools (or +/-tools) to run")

	flagN         = flag.Bool("n", false, "dry run")
	flagXedPath   = sgutil.FlagXEDPath(".")
	flagArm64Path = sgutil.FlagARM64Path(".")

	genFlags = gentools.RegisterFlags(nil)
)

// ToolSet is a [flag.Value] that accepts a comma-separated list of tool names.
// It rejects any tool names that aren't in the map. A list like "a,c" sets only
// "a" and "c" to true and all other tools to false. Alternatively, the list
// items may each start with + or -, which enables or disables (respectively)
// only the named tools.
type ToolSet map[string]bool

func (s ToolSet) String() string {
	var have []string
	for k, v := range s {
		if v {
			have = append(have, k)
		}
	}
	slices.Sort(have)
	return strings.Join(have, ",")
}

func (s ToolSet) Set(list string) error {
	list = strings.TrimSpace(list)
	isDelta := len(list) > 0 && (list[0] == '+' || list[0] == '-')
	if !isDelta {
		for k := range s {
			s[k] = false
		}
	}
	for item := range strings.SplitSeq(list, ",") {
		item = strings.TrimSpace(item)
		if item == "" {
			continue
		}
		var itemDelta bool
		val := true
		switch item[0] {
		case '+':
			val, itemDelta, item = true, true, item[1:]
		case '-':
			val, itemDelta, item = false, true, item[1:]
		}
		if isDelta != itemDelta {
			return fmt.Errorf("tool list %q mixes +/- and regular items", list)
		}
		if _, ok := s[item]; !ok {
			all := slices.Sorted(maps.Keys(s))
			return fmt.Errorf("unknown tool %s; valid tools are: %s", item, strings.Join(all, ", "))
		}
		s[item] = val
	}
	return nil
}

func flagVar[T flag.Value](name string, value T, usage string) T {
	flag.Var(value, name, usage)
	return value
}

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
	if flagTools["simdgen"] {
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

	if flagTools["simdgen"] {
		fmt.Fprintln(os.Stderr, "# This may take a few minutes...")
	}

	var files gentools.Files
	defer files.FlushOrExit()

	if flagTools["tmplgen"] {
		doGen("tmplgen", &files)
	}

	if flagTools["wasmgen"] || flagTools["simdgen"] {
		ssaGenPath := prettyPath(".", genFlags.OutputPath("cmd/compile/internal/ssa/_gen"))

		// If there is garbage in ssa/_gen/simdgenericOps.go, it can affect the merge in simdgen/wasmgen.
		if genFlags.Write {
			removeSimdGenericOps(ssaGenPath)
		}

		if flagTools["wasmgen"] {
			doGen("wasmgen", &files)
		}
		if flagTools["simdgen"] {
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

	if flagTools["midway"] {
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
