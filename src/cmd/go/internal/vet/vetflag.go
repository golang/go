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

// go vet flag processing
//
// We query the flags of the tool specified by -vettool and accept any
// of those flags plus any flag valid for 'go build'. The tool must
// support -flags, which prints a description of its flags in JSON to
// stdout.

// vetTool specifies the vet command to run.
// Any tool that supports the (still unpublished) vet
// command-line protocol may be supplied; see
// golang.org/x/tools/go/analysis/unitchecker for one
// implementation. It is also used by tests.
//
// The default behavior (vetTool=="") runs 'go tool vet'.
var vetTool string // -vettool

func init() {
	// For now, we omit the -json flag for vet because we could plausibly
	// support -json specific to the vet command in the future (perhaps using
	// the same format as build -json).
	work.AddBuildFlags(CmdVet, work.OmitJSONFlag)
	CmdVet.Flag.StringVar(&vetTool, "vettool", "", "")
}

func parseVettoolFlag(args []string) {
	// Extract -vettool by ad hoc flag processing:
	// its value is needed even before we can declare
	// the flags available during main flag processing.
	for i, arg := range args {
		if arg == "-vettool" || arg == "--vettool" {
			if i+1 >= len(args) {
				log.Fatalf("%s requires a filename", arg)
			}
			vetTool = args[i+1]
			return
		} else if strings.HasPrefix(arg, "-vettool=") ||
			strings.HasPrefix(arg, "--vettool=") {
			vetTool = arg[strings.IndexByte(arg, '=')+1:]
			return
		}
	}
}

// vetFlags processes the command line, splitting it at the first non-flag
// into the list of flags and list of packages.
func vetFlags(args []string) (passToVet, packageNames []string) {
	parseVettoolFlag(args)

	// Query the vet command for its flags.
	var tool string
	if vetTool == "" {
		tool = base.Tool("vet")
	} else {
		var err error
		tool, err = filepath.Abs(vetTool)
		if err != nil {
			log.Fatal(err)
		}
	}
	out := new(bytes.Buffer)
	vetcmd := exec.Command(tool, "-flags")
	vetcmd.Stdout = out
	if err := vetcmd.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "go: can't execute %s -flags: %v\n", tool, err)
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

	// Add vet's flags to CmdVet.Flag.
	//
	// Some flags, in particular -tags and -v, are known to vet but
	// also defined as build flags. This works fine, so we omit duplicates here.
	// However some, like -x, are known to the build but not to vet.
	isVetFlag := make(map[string]bool, len(analysisFlags))
	cf := CmdVet.Flag
	for _, f := range analysisFlags {
		isVetFlag[f.Name] = true
		if cf.Lookup(f.Name) == nil {
			if f.Bool {
				cf.Bool(f.Name, false, "")
			} else {
				cf.String(f.Name, "", "")
			}
		}
	}

	// Record the set of vet tool flags set by GOFLAGS. We want to pass them to
	// the vet tool, but only if they aren't overridden by an explicit argument.
	base.SetFromGOFLAGS(&CmdVet.Flag)
	addFromGOFLAGS := map[string]bool{}
	CmdVet.Flag.Visit(func(f *flag.Flag) {
		if isVetFlag[f.Name] {
			addFromGOFLAGS[f.Name] = true
		}
	})

	explicitFlags := make([]string, 0, len(args))
	for len(args) > 0 {
		f, remainingArgs, err := cmdflag.ParseOne(&CmdVet.Flag, args)

		if errors.Is(err, flag.ErrHelp) {
			exitWithUsage()
		}

		if errors.Is(err, cmdflag.ErrFlagTerminator) {
			// All remaining args must be package names, but the flag terminator is
			// not included.
			packageNames = remainingArgs
			break
		}

		if nf := (cmdflag.NonFlagError{}); errors.As(err, &nf) {
			// Everything from here on out — including the argument we just consumed —
			// must be a package name.
			packageNames = args
			break
		}

		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			exitWithUsage()
		}

		if isVetFlag[f.Name] {
			// Forward the raw arguments rather than cleaned equivalents, just in
			// case the vet tool parses them idiosyncratically.
			explicitFlags = append(explicitFlags, args[:len(args)-len(remainingArgs)]...)

			// This flag has been overridden explicitly, so don't forward its implicit
			// value from GOFLAGS.
			delete(addFromGOFLAGS, f.Name)
		}

		args = remainingArgs
	}

	// Prepend arguments from GOFLAGS before other arguments.
	CmdVet.Flag.Visit(func(f *flag.Flag) {
		if addFromGOFLAGS[f.Name] {
			passToVet = append(passToVet, fmt.Sprintf("-%s=%s", f.Name, f.Value))
		}
	})
	passToVet = append(passToVet, explicitFlags...)
	return passToVet, packageNames
}

func exitWithUsage() {
	fmt.Fprintf(os.Stderr, "usage: %s\n", CmdVet.UsageLine)
	fmt.Fprintf(os.Stderr, "Run 'go help %s' for details.\n", CmdVet.LongName())

	// This part is additional to what (*Command).Usage does:
	cmd := "go tool vet"
	if vetTool != "" {
		cmd = vetTool
	}
	fmt.Fprintf(os.Stderr, "Run '%s help' for a full list of flags and analyzers.\n", cmd)
	fmt.Fprintf(os.Stderr, "Run '%s -help' for an overview.\n", cmd)

	base.SetExitStatus(2)
	base.Exit()
}
