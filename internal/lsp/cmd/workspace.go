// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmd

import (
	"context"
	"flag"
	"fmt"

	"golang.org/x/tools/internal/lsp/command"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
)

// workspace is a top-level command for working with the gopls workspace. This
// is experimental and subject to change. The idea is that subcommands could be
// used for manipulating the workspace mod file, rather than editing it
// manually.
type workspace struct {
	subcommands
}

func newWorkspace(app *Application) *workspace {
	return &workspace{
		subcommands: subcommands{
			&generateWorkspaceMod{app: app},
		},
	}
}

func (w *workspace) Name() string { return "workspace" }
func (w *workspace) ShortHelp() string {
	return "manage the gopls workspace (experimental: under development)"
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
	cmd, err := command.NewGenerateGoplsModCommand("", command.URIArg{})
	if err != nil {
		return err
	}
	params := &protocol.ExecuteCommandParams{Command: cmd.Command, Arguments: cmd.Arguments}
	if _, err := conn.ExecuteCommand(ctx, params); err != nil {
		return fmt.Errorf("executing server command: %v", err)
	}
	return nil
}
