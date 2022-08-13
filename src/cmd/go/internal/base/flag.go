// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package base

import (
	"flag"

	"cmd/go/internal/cfg"
	"cmd/go/internal/fsys"
	"cmd/internal/quoted"
)

// A StringsFlag is a command-line flag that interprets its argument
// as a space-separated list of possibly-quoted strings.
type StringsFlag []string

func (v *StringsFlag) Set(s string) error {
	var err error
	*v, err = quoted.Split(s)
	if *v == nil {
		*v = []string{}
	}
	return err
}

func (v *StringsFlag) String() string {
	return "<StringsFlag>"
}

// explicitStringFlag is like a regular string flag, but it also tracks whether
// the string was set explicitly to a non-empty value.
type explicitStringFlag struct {
	value    *string
	explicit *bool
}

func (f explicitStringFlag) String() string {
	if f.value == nil {
		return ""
	}
	return *f.value
}

func (f explicitStringFlag) Set(v string) error {
	*f.value = v
	if v != "" {
		*f.explicit = true
	}
	return nil
}

// AddBuildFlagsNX adds the -n and -x build flags to the flag set.
func AddBuildFlagsNX(flags *flag.FlagSet) {
	flags.BoolVar(&cfg.BuildN, "n", false, "")
	flags.BoolVar(&cfg.BuildX, "x", false, "")
}

// AddModFlag adds the -mod build flag to the flag set.
func AddModFlag(flags *flag.FlagSet) {
	flags.Var(explicitStringFlag{value: &cfg.BuildMod, explicit: &cfg.BuildModExplicit}, "mod", "")
}

// AddModCommonFlags adds the module-related flags common to build commands
// and 'go mod' subcommands.
func AddModCommonFlags(flags *flag.FlagSet) {
	flags.BoolVar(&cfg.ModCacheRW, "modcacherw", false, "")
	flags.StringVar(&cfg.ModFile, "modfile", "", "")
	flags.StringVar(&fsys.OverlayFile, "overlay", "", "")
}
