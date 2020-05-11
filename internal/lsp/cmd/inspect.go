// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmd

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"

	"golang.org/x/tools/internal/lsp/lsprpc"
	"golang.org/x/tools/internal/tool"
)

type inspect struct {
	app *Application
}

func (i *inspect) subCommands() []tool.Application {
	return []tool.Application{
		&listSessions{app: i.app},
	}
}

func (i *inspect) Name() string  { return "inspect" }
func (i *inspect) Usage() string { return "<subcommand> [args...]" }
func (i *inspect) ShortHelp() string {
	return "inspect server state (daemon mode only)"
}
func (i *inspect) DetailedHelp(f *flag.FlagSet) {
	fmt.Fprint(f.Output(), "\nsubcommands:\n")
	for _, c := range i.subCommands() {
		fmt.Fprintf(f.Output(), "  %s: %s\n", c.Name(), c.ShortHelp())
	}
	f.PrintDefaults()
}

func (i *inspect) Run(ctx context.Context, args ...string) error {
	if len(args) == 0 {
		return tool.CommandLineErrorf("must provide subcommand to %q", i.Name())
	}
	command, args := args[0], args[1:]
	for _, c := range i.subCommands() {
		if c.Name() == command {
			return tool.Run(ctx, c, args)
		}
	}
	return tool.CommandLineErrorf("unknown command %v", command)
}

// listSessions is an inspect subcommand to list current sessions.
type listSessions struct {
	app *Application
}

func (c *listSessions) Name() string  { return "sessions" }
func (c *listSessions) Usage() string { return "" }
func (c *listSessions) ShortHelp() string {
	return "print information about current gopls sessions"
}

const listSessionsExamples = `
Examples:

1) list sessions for the default daemon:

$ gopls -remote=auto inspect sessions
or just
$ gopls inspect sessions

2) list sessions for a specific daemon:

$ gopls -remote=localhost:8082 inspect sessions
`

func (c *listSessions) DetailedHelp(f *flag.FlagSet) {
	fmt.Fprint(f.Output(), listSessionsExamples)
	f.PrintDefaults()
}

func (c *listSessions) Run(ctx context.Context, args ...string) error {
	remote := c.app.Remote
	if remote == "" {
		remote = "auto"
	}
	network, address := parseAddr(remote)
	state, err := lsprpc.QueryServerState(ctx, network, address)
	if err != nil {
		return err
	}
	v, err := json.MarshalIndent(state, "", "\t")
	if err != nil {
		log.Fatal(err)
	}
	os.Stdout.Write(v)
	return nil
}
