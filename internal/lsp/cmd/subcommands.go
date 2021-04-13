// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmd

import (
	"context"
	"flag"
	"fmt"

	"golang.org/x/tools/internal/tool"
)

// subcommands is a helper that may be embedded for commands that delegate to
// subcommands.
type subcommands []tool.Application

func (s subcommands) DetailedHelp(f *flag.FlagSet) {
	fmt.Fprint(f.Output(), "\nsubcommands:\n")
	for _, c := range s {
		fmt.Fprintf(f.Output(), "  %s: %s\n", c.Name(), c.ShortHelp())
	}
	f.PrintDefaults()
}

func (s subcommands) Usage() string { return "<subcommand> [args...]" }

func (s subcommands) Run(ctx context.Context, args ...string) error {
	if len(args) == 0 {
		return tool.CommandLineErrorf("must provide subcommand")
	}
	command, args := args[0], args[1:]
	for _, c := range s {
		if c.Name() == command {
			return tool.Run(ctx, c, args)
		}
	}
	return tool.CommandLineErrorf("unknown subcommand %v", command)
}
