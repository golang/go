// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vet

import (
	"flag"
	"fmt"
	"os"
	"strings"

	"cmd/go/internal/base"
	"cmd/go/internal/cmdflag"
	"cmd/go/internal/str"
	"cmd/go/internal/work"
)

const cmd = "vet"

// vetFlagDefn is the set of flags we process.
var vetFlagDefn = []*cmdflag.Defn{
	// Note: Some flags, in particular -tags and -v, are known to
	// vet but also defined as build flags. This works fine, so we
	// don't define them here but use AddBuildFlags to init them.
	// However some, like -x, are known to the build but not
	// to vet. We handle them in vetFlags.

	// local.
	{Name: "all", BoolVar: new(bool), PassToTest: true},
	{Name: "asmdecl", BoolVar: new(bool), PassToTest: true},
	{Name: "assign", BoolVar: new(bool), PassToTest: true},
	{Name: "atomic", BoolVar: new(bool), PassToTest: true},
	{Name: "bool", BoolVar: new(bool), PassToTest: true},
	{Name: "buildtags", BoolVar: new(bool), PassToTest: true},
	{Name: "cgocall", BoolVar: new(bool), PassToTest: true},
	{Name: "composites", BoolVar: new(bool), PassToTest: true},
	{Name: "copylocks", BoolVar: new(bool), PassToTest: true},
	{Name: "httpresponse", BoolVar: new(bool), PassToTest: true},
	{Name: "lostcancel", BoolVar: new(bool), PassToTest: true},
	{Name: "methods", BoolVar: new(bool), PassToTest: true},
	{Name: "nilfunc", BoolVar: new(bool), PassToTest: true},
	{Name: "printf", BoolVar: new(bool), PassToTest: true},
	{Name: "printfuncs", PassToTest: true},
	{Name: "rangeloops", BoolVar: new(bool), PassToTest: true},
	{Name: "shadow", BoolVar: new(bool), PassToTest: true},
	{Name: "shadowstrict", BoolVar: new(bool), PassToTest: true},
	{Name: "shift", BoolVar: new(bool), PassToTest: true},
	{Name: "source", BoolVar: new(bool), PassToTest: true},
	{Name: "structtags", BoolVar: new(bool), PassToTest: true},
	{Name: "tests", BoolVar: new(bool), PassToTest: true},
	{Name: "unreachable", BoolVar: new(bool), PassToTest: true},
	{Name: "unsafeptr", BoolVar: new(bool), PassToTest: true},
	{Name: "unusedfuncs", PassToTest: true},
	{Name: "unusedresult", BoolVar: new(bool), PassToTest: true},
	{Name: "unusedstringmethods", PassToTest: true},
}

var vetTool string

// add build flags to vetFlagDefn.
func init() {
	cmdflag.AddKnownFlags("vet", vetFlagDefn)
	var cmd base.Command
	work.AddBuildFlags(&cmd)
	cmd.Flag.StringVar(&vetTool, "vettool", "", "path to vet tool binary") // for cmd/vet tests; undocumented for now
	cmd.Flag.VisitAll(func(f *flag.Flag) {
		vetFlagDefn = append(vetFlagDefn, &cmdflag.Defn{
			Name:  f.Name,
			Value: f.Value,
		})
	})
}

// vetFlags processes the command line, splitting it at the first non-flag
// into the list of flags and list of packages.
func vetFlags(args []string) (passToVet, packageNames []string) {
	args = str.StringList(cmdflag.FindGOFLAGS(vetFlagDefn), args)
	for i := 0; i < len(args); i++ {
		if !strings.HasPrefix(args[i], "-") {
			return args[:i], args[i:]
		}

		f, value, extraWord := cmdflag.Parse(cmd, vetFlagDefn, args, i)
		if f == nil {
			fmt.Fprintf(os.Stderr, "vet: flag %q not defined\n", args[i])
			fmt.Fprintf(os.Stderr, "Run \"go help vet\" for more information\n")
			os.Exit(2)
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
