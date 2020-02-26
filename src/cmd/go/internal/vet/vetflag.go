// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vet

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"cmd/go/internal/base"
	"cmd/go/internal/cmdflag"
	"cmd/go/internal/str"
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
//
var vetTool string // -vettool

func init() {
	// Extract -vettool by ad hoc flag processing:
	// its value is needed even before we can declare
	// the flags available during main flag processing.
	for i, arg := range os.Args {
		if arg == "-vettool" || arg == "--vettool" {
			if i+1 >= len(os.Args) {
				log.Fatalf("%s requires a filename", arg)
			}
			vetTool = os.Args[i+1]
			break
		} else if strings.HasPrefix(arg, "-vettool=") ||
			strings.HasPrefix(arg, "--vettool=") {
			vetTool = arg[strings.IndexByte(arg, '=')+1:]
			break
		}
	}
}

// vetFlags processes the command line, splitting it at the first non-flag
// into the list of flags and list of packages.
func vetFlags(usage func(), args []string) (passToVet, packageNames []string) {
	// Query the vet command for its flags.
	tool := vetTool
	if tool != "" {
		var err error
		tool, err = filepath.Abs(tool)
		if err != nil {
			log.Fatal(err)
		}
	} else {
		tool = base.Tool("vet")
	}
	out := new(bytes.Buffer)
	vetcmd := exec.Command(tool, "-flags")
	vetcmd.Stdout = out
	if err := vetcmd.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "go vet: can't execute %s -flags: %v\n", tool, err)
		base.SetExitStatus(2)
		base.Exit()
	}
	var analysisFlags []struct {
		Name  string
		Bool  bool
		Usage string
	}
	if err := json.Unmarshal(out.Bytes(), &analysisFlags); err != nil {
		fmt.Fprintf(os.Stderr, "go vet: can't unmarshal JSON from %s -flags: %v", tool, err)
		base.SetExitStatus(2)
		base.Exit()
	}

	// Add vet's flags to vetflagDefn.
	//
	// Some flags, in particular -tags and -v, are known to vet but
	// also defined as build flags. This works fine, so we don't
	// define them here but use AddBuildFlags to init them.
	// However some, like -x, are known to the build but not to vet.
	var vetFlagDefn []*cmdflag.Defn
	for _, f := range analysisFlags {
		switch f.Name {
		case "tags", "v":
			continue
		}
		defn := &cmdflag.Defn{
			Name:       f.Name,
			PassToTest: true,
		}
		if f.Bool {
			defn.BoolVar = new(bool)
		}
		vetFlagDefn = append(vetFlagDefn, defn)
	}

	// Add build flags to vetFlagDefn.
	var cmd base.Command
	work.AddBuildFlags(&cmd, work.DefaultBuildFlags)
	// This flag declaration is a placeholder:
	// -vettool is actually parsed by the init function above.
	cmd.Flag.StringVar(new(string), "vettool", "", "path to vet tool binary")
	cmd.Flag.VisitAll(func(f *flag.Flag) {
		vetFlagDefn = append(vetFlagDefn, &cmdflag.Defn{
			Name:  f.Name,
			Value: f.Value,
		})
	})

	// Process args.
	goflags := cmdflag.FindGOFLAGS(vetFlagDefn)
	args = str.StringList(goflags, args)
	for i := 0; i < len(args); i++ {
		if !strings.HasPrefix(args[i], "-") {
			return args[:i], args[i:]
		}

		f, value, extraWord := cmdflag.Parse("vet", usage, vetFlagDefn, args, i)
		if f == nil {
			fmt.Fprintf(os.Stderr, "vet: flag %q not defined\n", args[i])
			fmt.Fprintf(os.Stderr, "Run \"go help vet\" for more information\n")
			base.SetExitStatus(2)
			base.Exit()
		}
		if i < len(goflags) {
			f.Present = false // Not actually present on the command line.
		}
		if f.Value != nil {
			if err := f.Value.Set(value); err != nil {
				base.Fatalf("invalid flag argument for -%s: %v", f.Name, err)
			}
			keep := f.PassToTest
			if !keep {
				// A build flag, probably one we don't want to pass to vet.
				// Can whitelist.
				switch f.Name {
				case "tags", "v":
					keep = true
				}
			}
			if !keep {
				// Flags known to the build but not to vet, so must be dropped.
				if extraWord {
					args = append(args[:i], args[i+2:]...)
					extraWord = false
				} else {
					args = append(args[:i], args[i+1:]...)
				}
				i--
			}
		}
		if extraWord {
			i++
		}
	}
	return args, nil
}

var vetUsage func()

func init() { vetUsage = usage } // break initialization cycle

func usage() {
	fmt.Fprintf(os.Stderr, "usage: %s\n", CmdVet.UsageLine)
	fmt.Fprintf(os.Stderr, "Run 'go help %s' for details.\n", CmdVet.LongName())

	// This part is additional to what (*Command).Usage does:
	cmd := "go tool vet"
	if vetTool != "" {
		cmd = vetTool
	}
	fmt.Fprintf(os.Stderr, "Run '%s -help' for the vet tool's flags.\n", cmd)

	base.SetExitStatus(2)
	base.Exit()
}
