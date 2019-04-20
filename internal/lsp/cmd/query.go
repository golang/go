// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmd

import (
	"context"
	"flag"
	"fmt"

	"golang.org/x/tools/internal/tool"
)

const (
	// The set of possible options that can be passed through the -emulate flag,
	// which causes query to adjust its output to match that of the binary being
	// emulated.

	// emulateGuru tells query to emulate the output format of the guru tool.
	emulateGuru = "guru"
)

// query implements the query command.
type query struct {
	JSON    bool   `flag:"json" help:"emit output in JSON format"`
	Emulate string `flag:"emulate" help:"compatibility mode, causes gopls to emulate another tool.\nvalues depend on the operation being performed"`

	app *Application
}

func (q *query) Name() string  { return "query" }
func (q *query) Usage() string { return "<mode> <mode args>" }
func (q *query) ShortHelp() string {
	return "answer queries about go source code"
}
func (q *query) DetailedHelp(f *flag.FlagSet) {
	fmt.Fprint(f.Output(), `
The mode argument determines the query to perform:
`)
	for _, m := range q.modes() {
		fmt.Fprintf(f.Output(), "  %s : %v\n", m.Name(), m.ShortHelp())
	}
	fmt.Fprint(f.Output(), `
query flags are:
`)
	f.PrintDefaults()
}

// Run takes the args after command flag processing, and invokes the correct
// query mode as specified by the first argument.
func (q *query) Run(ctx context.Context, args ...string) error {
	if len(args) == 0 {
		return tool.CommandLineErrorf("query must be supplied a mode")
	}
	mode, args := args[0], args[1:]
	for _, m := range q.modes() {
		if m.Name() == mode {
			tool.Main(ctx, m, args)
			return nil
		}
	}
	return tool.CommandLineErrorf("unknown command %v", mode)
}

// modes returns the set of modes supported by the query command.
func (q *query) modes() []tool.Application {
	return []tool.Application{
		&definition{query: q},
	}
}
