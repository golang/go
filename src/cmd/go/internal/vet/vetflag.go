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
	{Name: "all", BoolVar: new(bool)},
	{Name: "asmdecl", BoolVar: new(bool)},
	{Name: "assign", BoolVar: new(bool)},
	{Name: "atomic", BoolVar: new(bool)},
	{Name: "bool", BoolVar: new(bool)},
	{Name: "buildtags", BoolVar: new(bool)},
	{Name: "cgocall", BoolVar: new(bool)},
	{Name: "composites", BoolVar: new(bool)},
	{Name: "copylocks", BoolVar: new(bool)},
	{Name: "httpresponse", BoolVar: new(bool)},
	{Name: "lostcancel", BoolVar: new(bool)},
	{Name: "methods", BoolVar: new(bool)},
	{Name: "nilfunc", BoolVar: new(bool)},
	{Name: "printf", BoolVar: new(bool)},
	{Name: "printfuncs"},
	{Name: "rangeloops", BoolVar: new(bool)},
	{Name: "shadow", BoolVar: new(bool)},
	{Name: "shadowstrict", BoolVar: new(bool)},
	{Name: "shift", BoolVar: new(bool)},
	{Name: "source", BoolVar: new(bool)},
	{Name: "structtags", BoolVar: new(bool)},
	{Name: "tests", BoolVar: new(bool)},
	{Name: "unreachable", BoolVar: new(bool)},
	{Name: "unsafeptr", BoolVar: new(bool)},
	{Name: "unusedfuncs"},
	{Name: "unusedresult", BoolVar: new(bool)},
	{Name: "unusedstringmethods"},
}

var vetTool string

// add build flags to vetFlagDefn.
func init() {
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
			switch f.Name {
			// Flags known to the build but not to vet, so must be dropped.
			case "x", "n", "vettool", "compiler":
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
