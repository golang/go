// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package logos implements logos-mode setup for the go command.
package logos

import (
	"strconv"
	"strings"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/logos/contextcmd"
	"cmd/go/internal/logos/eject"
	"cmd/go/internal/logos/lintcmd"
	"cmd/go/internal/logos/lsp"
	"cmd/go/internal/logos/next"
)

// ConfigureCommand adjusts the root command for logos mode.
func ConfigureCommand(cmd *base.Command) {
	cmd.UsageLine = "logos"
	cmd.Long = "Logos is a tool for managing loGo/s source code."
	dropFmt(cmd)
	for _, sub := range cmd.Commands {
		rebrandUsageLine(sub)
	}
	contextcmd.Register(cmd)
	eject.Register(cmd)
	lintcmd.Register(cmd)
	lsp.Register(cmd)
	next.Register(cmd)
	addPedanticFlag(cmd.Lookup("build"))
	addPedanticFlag(cmd.Lookup("lint"))
	addJSONDiagnosticsFlag(cmd.Lookup("build"))
	addJSONDiagnosticsFlag(cmd.Lookup("lint"))
	addWasmLoaderHelp(cmd.Lookup("build"))
	addGCFlagHelp(cmd.Lookup("build"))
}

// pedanticFlag is the --pedantic flag value: a boolean that also records
// that it was given, so --pedantic=false overrides a logos.toml pedantic
// key and an absent flag does not.
type pedanticFlag struct{}

// IsBoolFlag lets --pedantic stand without a value.
func (pedanticFlag) IsBoolFlag() bool { return true }

// String returns the current value.
func (pedanticFlag) String() string { return strconv.FormatBool(cfg.Pedantic) }

// Set parses a boolean and records that the flag was set.
func (pedanticFlag) Set(s string) error {
	v, err := strconv.ParseBool(s)
	if err != nil {
		return err
	}
	cfg.Pedantic, cfg.PedanticSet = v, true
	return nil
}

// addPedanticFlag registers the logos-only --pedantic flag on cmd, a
// logos build or logos lint command, and documents it in its help. It runs
// only from ConfigureCommand, so plain "go build" keeps the upstream
// unknown-flag error and "go help build" keeps the upstream text.
func addPedanticFlag(cmd *base.Command) {
	if cmd == nil {
		return
	}
	if cmd.Flag.Lookup("pedantic") == nil {
		cmd.Flag.Var(pedanticFlag{}, "pedantic", "")
	}
	addHelp(cmd, "-pedantic", "\n\t-pedantic\n\t\tenable pedantic diagnostics (logos builds only).\n")
}

// addJSONDiagnosticsFlag registers the logos-only --json-diagnostics flag
// on cmd and documents it in its help. Like --pedantic it exists only in
// logos mode.
func addJSONDiagnosticsFlag(cmd *base.Command) {
	if cmd == nil {
		return
	}
	if cmd.Flag.Lookup("json-diagnostics") == nil {
		cmd.Flag.BoolVar(&cfg.LogosJSON, "json-diagnostics", false, "")
	}
	addHelp(cmd, "-json-diagnostics", "\n\t-json-diagnostics\n\t\tprint each compiler and lint diagnostic as one JSON object\n\t\tper line on stderr (logos builds only).\n")
}

// addGCFlagHelp documents the -gc flag in the logos help of cmd. The flag is
// registered for every build-flag command under plain go too, where it has no
// help text.
func addGCFlagHelp(cmd *base.Command) {
	if cmd == nil {
		return
	}
	addHelp(cmd, "-gc mode", "\n\t-gc mode\n\t\tselect the garbage collector: greentea, zgc, immix, wasm,\n\t\tbatch, or epsilon. It overrides the gc key of logos.toml.\n\t\tThe default is greentea, or wasm on GOARCH=wasm.\n")
}

// addWasmLoaderHelp documents the --wasm-loader flag in the logos help of
// cmd. The flag itself is registered with the shared build flags, so plain
// go accepts it too, but only logos help names it: "go help build" keeps
// the upstream text.
func addWasmLoaderHelp(cmd *base.Command) {
	if cmd == nil {
		return
	}
	addHelp(cmd, "-wasm-loader", "\n\t-wasm-loader\n\t\twrite <name>.loader.mjs, the wasm host loader, beside the output of a\n\t\tGOARCH=wasm main package when -o names a file with a .wasm suffix.\n\t\tThe default is true; -wasm-loader=false turns it off.\n")
}

// addHelp appends a flag description to the long help of cmd before its
// "See also:" line, unless the help already names flag.

// addHelp appends a flag description to the long help of cmd before its
// "See also:" line, unless the help already names flag.
func addHelp(cmd *base.Command, flag, help string) {
	if strings.Contains(cmd.Long, flag) {
		return
	}
	before, after, found := strings.Cut(cmd.Long, "\nSee also:")
	if found {
		cmd.Long = before + help + "\nSee also:" + after
		return
	}
	cmd.Long += help
}

// rebrandUsageLine swaps the leading go verb for logos in one command's
// usage line, recursing into subcommands. Resolution is unaffected:
// base strips either verb when computing long names.
func rebrandUsageLine(cmd *base.Command) {
	if rest, ok := strings.CutPrefix(cmd.UsageLine, "go "); ok {
		cmd.UsageLine = "logos " + rest
	}
	for _, sub := range cmd.Commands {
		rebrandUsageLine(sub)
	}
}

// ProgramName returns the current program name for user-facing text.
func ProgramName() string {
	if cfg.Logos {
		return "logos"
	}
	return "go"
}
