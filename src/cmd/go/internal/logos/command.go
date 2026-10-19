// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package logos implements logos-mode setup for the go command.
package logos

import (
	"strings"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/logos/eject"
)

// ConfigureCommand adjusts the root command for logos mode.
func ConfigureCommand(cmd *base.Command) {
	cmd.UsageLine = "logos"
	cmd.Long = "Logos is a tool for managing loGo/s source code."
	for _, sub := range cmd.Commands {
		rebrandUsageLine(sub)
	}
	addPedanticFlag(cmd)
	eject.Register(cmd)
}

// addPedanticFlag registers the logos-only --pedantic build flag on
// "logos build" and documents it in "logos help build". It runs only
// from ConfigureCommand, so plain "go build" keeps the upstream
// unknown-flag error and "go help build" keeps the upstream text.
func addPedanticFlag(cmd *base.Command) {
	build := cmd.Lookup("build")
	if build == nil {
		return
	}
	if build.Flag.Lookup("pedantic") == nil {
		build.Flag.BoolVar(&cfg.Pedantic, "pedantic", false, "")
	}
	if strings.Contains(build.Long, "-pedantic") {
		return
	}
	const help = "\n\t-pedantic\n\t\tenable pedantic diagnostics (logos builds only).\n"
	before, after, found := strings.Cut(build.Long, "\nSee also:")
	if found {
		build.Long = before + help + "\nSee also:" + after
		return
	}
	build.Long += help
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
