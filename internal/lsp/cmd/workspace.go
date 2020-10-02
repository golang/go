// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmd

import (
	"context"
	"flag"
	"fmt"

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/tool"
)

// workspace is a top-level command for working with the gopls workspace. This
// is experimental and subject to change. The idea is that subcommands could be
// used for manipulating the workspace mod file, rather than editing it
// manually.
type workspace struct {
	app *Application
}

func (w *workspace) subCommands() []tool.Application {
	return []tool.Application{
		&generateWorkspaceMod{app: w.app},
	}
}

func (w *workspace) Name() string  { return "workspace" }
func (w *workspace) Usage() string { return "<subcommand> [args...]" }
func (w *workspace) ShortHelp() string {
	return "manage the gopls workspace (experimental: under development)"
}

func (w *workspace) DetailedHelp(f *flag.FlagSet) {
	fmt.Fprint(f.Output(), "\nsubcommands:\n")
	for _, c := range w.subCommands() {
		fmt.Fprintf(f.Output(), "  %s: %s\n", c.Name(), c.ShortHelp())
	}
	f.PrintDefaults()
}

func (w *workspace) Run(ctx context.Context, args ...string) error {
	if len(args) == 0 {
		return tool.CommandLineErrorf("must provide subcommand to %q", w.Name())
	}
	command, args := args[0], args[1:]
	for _, c := range w.subCommands() {
		if c.Name() == command {
			return tool.Run(ctx, c, args)
		}
	}
	return tool.CommandLineErrorf("unknown command %v", command)
}

// generateWorkspaceMod (re)generates the gopls.mod file for the current
// workspace.
type generateWorkspaceMod struct {
	app *Application
}

func (c *generateWorkspaceMod) Name() string  { return "generate" }
func (c *generateWorkspaceMod) Usage() string { return "" }
func (c *generateWorkspaceMod) ShortHelp() string {
	return "generate a gopls.mod file for a workspace"
}

func (c *generateWorkspaceMod) DetailedHelp(f *flag.FlagSet) {
	f.PrintDefaults()
}

func (c *generateWorkspaceMod) Run(ctx context.Context, args ...string) error {
	origOptions := c.app.options
	c.app.options = func(opts *source.Options) {
		origOptions(opts)
		opts.ExperimentalWorkspaceModule = true
	}
	conn, err := c.app.connect(ctx)
	if err != nil {
		return err
	}
	defer conn.terminate(ctx)
	params := &protocol.ExecuteCommandParams{Command: source.CommandGenerateGoplsMod.ID()}
	if _, err := conn.ExecuteCommand(ctx, params); err != nil {
		return fmt.Errorf("executing server command: %v", err)
	}
	return nil
}
