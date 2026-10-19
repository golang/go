// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package logos implements logos-mode setup for the go command.
package logos

import (
	"strings"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
)

// ConfigureCommand adjusts the root command for logos mode.
func ConfigureCommand(cmd *base.Command) {
	cmd.UsageLine = "logos"
	cmd.Long = "Logos is a tool for managing loGo/s source code."
	for _, sub := range cmd.Commands {
		rebrandUsageLine(sub)
	}
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
