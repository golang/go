// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmd

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"golang.org/x/tools/internal/lsp/command"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/tool"
)

// vulncheck implements the vulncheck command.
type vulncheck struct {
	app *Application
}

func (v *vulncheck) Name() string   { return "vulncheck" }
func (v *vulncheck) Parent() string { return v.app.Name() }
func (v *vulncheck) Usage() string  { return "" }
func (v *vulncheck) ShortHelp() string {
	return "run experimental vulncheck analysis (experimental: under development)"
}
func (v *vulncheck) DetailedHelp(f *flag.FlagSet) {
	fmt.Fprint(f.Output(), `
	WARNING: this command is experimental.

	Example:
	$ gopls vulncheck <packages>
`)
	printFlagDefaults(f)
}

func (v *vulncheck) Run(ctx context.Context, args ...string) error {
	if len(args) > 1 {
		return tool.CommandLineErrorf("vulncheck accepts at most one package pattern")
	}
	pattern := "."
	if len(args) == 1 {
		pattern = args[0]
	}

	conn, err := v.app.connect(ctx)
	if err != nil {
		return err
	}
	defer conn.terminate(ctx)

	cwd, err := os.Getwd()
	if err != nil {
		return err
	}

	cmd, err := command.NewRunVulncheckExpCommand("", command.VulncheckArgs{
		Dir:     protocol.URIFromPath(cwd),
		Pattern: pattern,
	})
	if err != nil {
		return err
	}

	params := &protocol.ExecuteCommandParams{Command: cmd.Command, Arguments: cmd.Arguments}
	res, err := conn.ExecuteCommand(ctx, params)
	if err != nil {
		return fmt.Errorf("executing server command: %v", err)
	}
	data, err := json.MarshalIndent(res, " ", " ")
	if err != nil {
		return fmt.Errorf("failed to decode results: %v", err)
	}
	fmt.Printf("%s\n", data)
	return nil
}
