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

	"golang.org/x/tools/go/packages"
	vulnchecklib "golang.org/x/tools/gopls/internal/vulncheck"
	"golang.org/x/tools/internal/tool"
)

// vulncheck implements the vulncheck command.
type vulncheck struct {
	Config    bool `flag:"config" help:"If true, the command reads a JSON-encoded package load configuration from stdin"`
	AsSummary bool `flag:"summary" help:"If true, outputs a JSON-encoded govulnchecklib.Summary JSON"`
	app       *Application
}

type pkgLoadConfig struct {
	// BuildFlags is a list of command-line flags to be passed through to
	// the build system's query tool.
	BuildFlags []string

	// If Tests is set, the loader includes related test packages.
	Tests bool
}

// TODO(hyangah): document pkgLoadConfig

func (v *vulncheck) Name() string   { return "vulncheck" }
func (v *vulncheck) Parent() string { return v.app.Name() }
func (v *vulncheck) Usage() string  { return "" }
func (v *vulncheck) ShortHelp() string {
	return "run experimental vulncheck analysis (experimental: under development)"
}
func (v *vulncheck) DetailedHelp(f *flag.FlagSet) {
	fmt.Fprint(f.Output(), `
	WARNING: this command is experimental.

	By default, the command outputs a JSON-encoded
	golang.org/x/tools/gopls/internal/lsp/command.VulncheckResult
	message.
	Example:
	$ gopls vulncheck <packages>

`)
	printFlagDefaults(f)
}

func (v *vulncheck) Run(ctx context.Context, args ...string) error {
	if vulnchecklib.Main == nil {
		return fmt.Errorf("vulncheck command is available only in gopls compiled with go1.18 or newer")
	}

	// TODO(hyangah): what's wrong with allowing multiple targets?
	if len(args) > 1 {
		return tool.CommandLineErrorf("vulncheck accepts at most one package pattern")
	}
	var cfg pkgLoadConfig
	if v.Config {
		if err := json.NewDecoder(os.Stdin).Decode(&cfg); err != nil {
			return tool.CommandLineErrorf("failed to parse cfg: %v", err)
		}
	}
	loadCfg := packages.Config{
		Context:    ctx,
		Tests:      cfg.Tests,
		BuildFlags: cfg.BuildFlags,
		// inherit the current process's cwd and env.
	}

	if err := vulnchecklib.Main(loadCfg, args...); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	return nil
}
