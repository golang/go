// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmd

import (
	"context"
	"flag"
	"fmt"
	"text/tabwriter"

	"golang.org/x/tools/internal/tool"
)

// subcommands is a helper that may be embedded for commands that delegate to
// subcommands.
type subcommands []tool.Application

func (s subcommands) DetailedHelp(f *flag.FlagSet) {
	w := tabwriter.NewWriter(f.Output(), 0, 0, 2, ' ', 0)
	defer w.Flush()
	fmt.Fprint(w, "\nSubcommand:\n")
	for _, c := range s {
		fmt.Fprintf(w, "  %s\t%s\n", c.Name(), c.ShortHelp())
	}
	printFlagDefaults(f)
}

func (s subcommands) Usage() string { return "<subcommand> [arg]..." }

func (s subcommands) Run(ctx context.Context, args ...string) error {
	if len(args) == 0 {
		return tool.CommandLineErrorf("must provide subcommand")
	}
	command, args := args[0], args[1:]
	for _, c := range s {
		if c.Name() == command {
			s := flag.NewFlagSet(c.Name(), flag.ExitOnError)
			return tool.Run(ctx, s, c, args)
		}
	}
	return tool.CommandLineErrorf("unknown subcommand %v", command)
}
