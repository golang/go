// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmd

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	"os"

	"golang.org/x/tools/gopls/internal/lsp/command"
	"golang.org/x/tools/gopls/internal/lsp/lsprpc"
)

type remote struct {
	app *Application
	subcommands

	// For backward compatibility, allow aliasing this command (it was previously
	// called 'inspect').
	//
	// TODO(rFindley): delete this after allowing some transition time in case
	//                 there were any users of 'inspect' (I suspect not).
	alias string
}

func newRemote(app *Application, alias string) *remote {
	return &remote{
		app: app,
		subcommands: subcommands{
			&listSessions{app: app},
			&startDebugging{app: app},
		},
		alias: alias,
	}
}

func (r *remote) Name() string {
	if r.alias != "" {
		return r.alias
	}
	return "remote"
}

func (r *remote) Parent() string { return r.app.Name() }

func (r *remote) ShortHelp() string {
	short := "interact with the gopls daemon"
	if r.alias != "" {
		short += " (deprecated: use 'remote')"
	}
	return short
}

// listSessions is an inspect subcommand to list current sessions.
type listSessions struct {
	app *Application
}

func (c *listSessions) Name() string   { return "sessions" }
func (c *listSessions) Parent() string { return c.app.Name() }
func (c *listSessions) Usage() string  { return "" }
func (c *listSessions) ShortHelp() string {
	return "print information about current gopls sessions"
}

const listSessionsExamples = `
Examples:

1) list sessions for the default daemon:

$ gopls -remote=auto remote sessions
or just
$ gopls remote sessions

2) list sessions for a specific daemon:

$ gopls -remote=localhost:8082 remote sessions
`

func (c *listSessions) DetailedHelp(f *flag.FlagSet) {
	fmt.Fprint(f.Output(), listSessionsExamples)
	printFlagDefaults(f)
}

func (c *listSessions) Run(ctx context.Context, args ...string) error {
	remote := c.app.Remote
	if remote == "" {
		remote = "auto"
	}
	state, err := lsprpc.QueryServerState(ctx, remote)
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

type startDebugging struct {
	app *Application
}

func (c *startDebugging) Name() string  { return "debug" }
func (c *startDebugging) Usage() string { return "[host:port]" }
func (c *startDebugging) ShortHelp() string {
	return "start the debug server"
}

const startDebuggingExamples = `
Examples:

1) start a debug server for the default daemon, on an arbitrary port:

$ gopls -remote=auto remote debug
or just
$ gopls remote debug

2) start for a specific daemon, on a specific port:

$ gopls -remote=localhost:8082 remote debug localhost:8083
`

func (c *startDebugging) DetailedHelp(f *flag.FlagSet) {
	fmt.Fprint(f.Output(), startDebuggingExamples)
	printFlagDefaults(f)
}

func (c *startDebugging) Run(ctx context.Context, args ...string) error {
	if len(args) > 1 {
		fmt.Fprintln(os.Stderr, c.Usage())
		return errors.New("invalid usage")
	}
	remote := c.app.Remote
	if remote == "" {
		remote = "auto"
	}
	debugAddr := ""
	if len(args) > 0 {
		debugAddr = args[0]
	}
	debugArgs := command.DebuggingArgs{
		Addr: debugAddr,
	}
	var result command.DebuggingResult
	if err := lsprpc.ExecuteCommand(ctx, remote, command.StartDebugging.ID(), debugArgs, &result); err != nil {
		return err
	}
	if len(result.URLs) == 0 {
		return errors.New("no debugging URLs")
	}
	for _, url := range result.URLs {
		fmt.Printf("debugging on %s\n", url)
	}
	return nil
}
