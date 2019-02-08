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
	"go/ast"
	"go/parser"
	"go/token"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/tool"
)

// Application is the main application as passed to tool.Main
// It handles the main command line parsing and dispatch to the sub commands.
type Application struct {
	// Core application flags

	// Embed the basic profiling flags supported by the tool package
	tool.Profile

	// We include the server configuration directly for now, so the flags work
	// even without the verb.
	// TODO: Remove this when we stop allowing the serve verb by default.
	Serve Serve

	// An initial, common go/packages configuration
	Config packages.Config

	// Support for remote lsp server
	Remote string `flag:"remote" help:"*EXPERIMENTAL* - forward all commands to a remote lsp"`
}

// Name implements tool.Application returning the binary name.
func (app *Application) Name() string { return "gopls" }

// Usage implements tool.Application returning empty extra argument usage.
func (app *Application) Usage() string { return "<command> [command-flags] [command-args]" }

// ShortHelp implements tool.Application returning the main binary help.
func (app *Application) ShortHelp() string {
	return "The Go Language source tools."
}

// DetailedHelp implements tool.Application returning the main binary help.
// This includes the short help for all the sub commands.
func (app *Application) DetailedHelp(f *flag.FlagSet) {
	fmt.Fprint(f.Output(), `
Available commands are:
`)
	for _, c := range app.commands() {
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
	app.Serve.app = app
	if len(args) == 0 {
		tool.Main(ctx, &app.Serve, args)
		return nil
	}
	app.Config.Mode = packages.LoadSyntax
	app.Config.Tests = true
	if app.Config.Fset == nil {
		app.Config.Fset = token.NewFileSet()
	}
	app.Config.Context = ctx
	app.Config.ParseFile = func(fset *token.FileSet, filename string, src []byte) (*ast.File, error) {
		return parser.ParseFile(fset, filename, src, parser.AllErrors|parser.ParseComments)
	}
	command, args := args[0], args[1:]
	for _, c := range app.commands() {
		if c.Name() == command {
			tool.Main(ctx, c, args)
			return nil
		}
	}
	return tool.CommandLineErrorf("Unknown command %v", command)
}

// commands returns the set of commands supported by the gopls tool on the
// command line.
// The command is specified by the first non flag argument.
func (app *Application) commands() []tool.Application {
	return []tool.Application{
		&app.Serve,
		&query{app: app},
	}
}
