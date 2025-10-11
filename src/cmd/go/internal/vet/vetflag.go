// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vet

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"cmd/go/internal/base"
	"cmd/go/internal/cmdflag"
	"cmd/go/internal/work"
)

// go vet/fix flag processing
var (
	// We query the flags of the tool specified by -{vet,fix}tool
	// and accept any of those flags plus any flag valid for 'go
	// build'. The tool must support -flags, which prints a
	// description of its flags in JSON to stdout.

	// toolFlag specifies the vet/fix command to run.
	// Any toolFlag that supports the (unpublished) vet
	// command-line protocol may be supplied; see
	// golang.org/x/tools/go/analysis/unitchecker for the
	// sole implementation. It is also used by tests.
	//
	// The default behavior ("") runs 'go tool {vet,fix}'.
	//
	// Do not access this flag directly; use [parseToolFlag].
	toolFlag    string // -{vet,fix}tool
	diffFlag    bool   // -diff
	jsonFlag    bool   // -json
	contextFlag = -1   // -c=n
)

func addFlags(cmd *base.Command) {
	// We run the compiler for export data.
	// Suppress the build -json flag; we define our own.
	work.AddBuildFlags(cmd, work.OmitJSONFlag)

	cmd.Flag.StringVar(&toolFlag, cmd.Name()+"tool", "", "") // -vettool or -fixtool
	cmd.Flag.BoolVar(&diffFlag, "diff", false, "print diff instead of applying it")
	cmd.Flag.BoolVar(&jsonFlag, "json", false, "print diagnostics and fixes as JSON")
	cmd.Flag.IntVar(&contextFlag, "c", -1, "display offending line with this many lines of context")
}

// parseToolFlag scans args for -{vet,fix}tool and returns the effective tool filename.
func parseToolFlag(cmd *base.Command, args []string) string {
	toolFlagName := cmd.Name() + "tool" // vettool or fixtool

	// Extract -{vet,fix}tool by ad hoc flag processing:
	// its value is needed even before we can declare
	// the flags available during main flag processing.
	for i, arg := range args {
		if arg == "-"+toolFlagName || arg == "--"+toolFlagName {
			if i+1 >= len(args) {
				log.Fatalf("%s requires a filename", arg)
			}
			toolFlag = args[i+1]
			break
		} else if strings.HasPrefix(arg, "-"+toolFlagName+"=") ||
			strings.HasPrefix(arg, "--"+toolFlagName+"=") {
			toolFlag = arg[strings.IndexByte(arg, '=')+1:]
			break
		}
	}

	if toolFlag != "" {
		tool, err := filepath.Abs(toolFlag)
		if err != nil {
			log.Fatal(err)
		}
		return tool
	}

	return base.Tool(cmd.Name()) // default to 'go tool vet|fix'
}

// toolFlags processes the command line, splitting it at the first non-flag
// into the list of flags and list of packages.
func toolFlags(cmd *base.Command, args []string) (passToTool, packageNames []string) {
	tool := parseToolFlag(cmd, args)
	work.VetTool = tool

	// Query the tool for its flags.
	out := new(bytes.Buffer)
	toolcmd := exec.Command(tool, "-flags")
	toolcmd.Stdout = out
	if err := toolcmd.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "go: %s -flags failed: %v\n", tool, err)
		base.SetExitStatus(2)
		base.Exit()
	}
	var analysisFlags []struct {
		Name  string
		Bool  bool
		Usage string
	}
	if err := json.Unmarshal(out.Bytes(), &analysisFlags); err != nil {
		fmt.Fprintf(os.Stderr, "go: can't unmarshal JSON from %s -flags: %v", tool, err)
		base.SetExitStatus(2)
		base.Exit()
	}

	// Add tool's flags to cmd.Flag.
	//
	// Some flags, in particular -tags and -v, are known to the tool but
	// also defined as build flags. This works fine, so we omit duplicates here.
	// However some, like -x, are known to the build but not to the tool.
	isToolFlag := make(map[string]bool, len(analysisFlags))
	cf := cmd.Flag
	for _, f := range analysisFlags {
		// We reimplement the unitchecker's -c=n flag.
		// Don't allow it to be passed through.
		if f.Name == "c" {
			continue
		}
		isToolFlag[f.Name] = true
		if cf.Lookup(f.Name) == nil {
			if f.Bool {
				cf.Bool(f.Name, false, "")
			} else {
				cf.String(f.Name, "", "")
			}
		}
	}

	// Record the set of tool flags set by GOFLAGS. We want to pass them to
	// the tool, but only if they aren't overridden by an explicit argument.
	base.SetFromGOFLAGS(&cmd.Flag)
	addFromGOFLAGS := map[string]bool{}
	cmd.Flag.Visit(func(f *flag.Flag) {
		if isToolFlag[f.Name] {
			addFromGOFLAGS[f.Name] = true
		}
	})

	explicitFlags := make([]string, 0, len(args))
	for len(args) > 0 {
		f, remainingArgs, err := cmdflag.ParseOne(&cmd.Flag, args)

		if errors.Is(err, flag.ErrHelp) {
			exitWithUsage(cmd)
		}

		if errors.Is(err, cmdflag.ErrFlagTerminator) {
			// All remaining args must be package names, but the flag terminator is
			// not included.
			packageNames = remainingArgs
			break
		}

		if _, ok := errors.AsType[cmdflag.NonFlagError](err); ok {
			// Everything from here on out — including the argument we just consumed —
			// must be a package name.
			packageNames = args
			break
		}

		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			exitWithUsage(cmd)
		}

		if isToolFlag[f.Name] {
			// Forward the raw arguments rather than cleaned equivalents, just in
			// case the tool parses them idiosyncratically.
			explicitFlags = append(explicitFlags, args[:len(args)-len(remainingArgs)]...)

			// This flag has been overridden explicitly, so don't forward its implicit
			// value from GOFLAGS.
			delete(addFromGOFLAGS, f.Name)
		}

		args = remainingArgs
	}

	// Prepend arguments from GOFLAGS before other arguments.
	cmd.Flag.Visit(func(f *flag.Flag) {
		if addFromGOFLAGS[f.Name] {
			passToTool = append(passToTool, fmt.Sprintf("-%s=%s", f.Name, f.Value))
		}
	})
	passToTool = append(passToTool, explicitFlags...)
	return passToTool, packageNames
}

func exitWithUsage(cmd *base.Command) {
	fmt.Fprintf(os.Stderr, "usage: %s\n", cmd.UsageLine)
	fmt.Fprintf(os.Stderr, "Run 'go help %s' for details.\n", cmd.LongName())

	// This part is additional to what (*Command).Usage does:
	tool := toolFlag
	if tool == "" {
		tool = "go tool " + cmd.Name()
	}
	fmt.Fprintf(os.Stderr, "Run '%s help' for a full list of flags and analyzers.\n", tool)
	fmt.Fprintf(os.Stderr, "Run '%s -help' for an overview.\n", tool)

	base.SetExitStatus(2)
	base.Exit()
}
