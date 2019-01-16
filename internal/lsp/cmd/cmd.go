// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package cmd handles the gopls command line.
// It contains a handler for each of the modes, along with all the flag handling
// and the command line output format.
package cmd

import (
	"context"
	"flag"
	"fmt"
	"golang.org/x/tools/internal/tool"
)

// Application is the main application as passed to tool.Main
// It handles the main command line parsing and dispatch to the sub commands.
type Application struct {
	// Embed the basic profiling flags supported by the tool package
	tool.Profile

	// we also include the server directly for now, so the flags work even without
	// the verb. We should remove this when we stop allowing the server verb by
	// default
	Server Server
}

// Name implements tool.Application returning the binary name.
func (app *Application) Name() string { return "gopls" }

// Usage implements tool.Application returning empty extra argument usage.
func (app *Application) Usage() string { return "<mode> [mode-flags] [mode-args]" }

// ShortHelp implements tool.Application returning the main binary help.
func (app *Application) ShortHelp() string {
	return "The Go Language Smartness Provider."
}

// DetailedHelp implements tool.Application returning the main binary help.
// This includes the short help for all the sub commands.
func (app *Application) DetailedHelp(f *flag.FlagSet) {
	fmt.Fprint(f.Output(), `
Available modes are:
`)
	for _, c := range app.modes() {
		fmt.Fprintf(f.Output(), "  %s : %v\n", c.Name(), c.ShortHelp())
	}
	fmt.Fprint(f.Output(), `
gopls flags are:
`)
	f.PrintDefaults()
}

// Run takes the args after top level flag processing, and invokes the correct
// sub command as specified by the first argument.
// If no arguments are passed it will invoke the server sub command, as a
// temporary measure for compatibility.
func (app *Application) Run(ctx context.Context, args ...string) error {
	if len(args) == 0 {
		tool.Main(ctx, &app.Server, args)
		return nil
	}
	mode, args := args[0], args[1:]
	for _, m := range app.modes() {
		if m.Name() == mode {
			tool.Main(ctx, m, args)
			return nil
		}
	}
	return tool.CommandLineErrorf("Unknown mode %v", mode)
}

// modes returns the set of command modes supported by the gopls tool on the
// command line.
// The mode is specified by the first non flag argument.
func (app *Application) modes() []tool.Application {
	return []tool.Application{
		&app.Server,
	}
}
