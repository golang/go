// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package base

import (
	"flag"

	"cmd/go/internal/cfg"
	"cmd/go/internal/str"
)

// A StringsFlag is a command-line flag that interprets its argument
// as a space-separated list of possibly-quoted strings.
type StringsFlag []string

func (v *StringsFlag) Set(s string) error {
	var err error
	*v, err = str.SplitQuotedFields(s)
	if *v == nil {
		*v = []string{}
	}
	return err
}

func (v *StringsFlag) String() string {
	return "<StringsFlag>"
}

// AddBuildFlagsNX adds the -n and -x build flags to the flag set.
func AddBuildFlagsNX(flags *flag.FlagSet) {
	flags.BoolVar(&cfg.BuildN, "n", false, "")
	flags.BoolVar(&cfg.BuildX, "x", false, "")
}

// AddLoadFlags adds the -mod build flag to the flag set.
func AddLoadFlags(flags *flag.FlagSet) {
	flags.StringVar(&cfg.BuildMod, "mod", "", "")
}
