// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package service implements the “go service” command.
package service

import (
	"cmd/internal/telemetry/counter"
	"context"

	"cmd/go/internal/base"
)

var CmdService = &base.Command{
	Run:         createServiceScafflod,
	UsageLine:   "go service [init] [service-name]",
	Short:       "create backend service scaffolding",
	CustomFlags: true,
	Long: `
	Service init creates the scaffolding of a minimal backend service with the given service name.`,
}

// Return whether tool can be expected in the gccgo tool directory.
// Other binaries could be in the same directory so don't
// show those with the 'go tool' command.
func isGccgoTool(tool string) bool {
	switch tool {
	case "cgo", "fix", "cover", "godoc", "vet":
		return true
	}
	return false
}

// createServiceScafflod creates the service scaffolding
func createServiceScafflod(ctx context.Context, cmd *base.Command, args []string) {
	if len(args) == 0 {
		counter.Inc("go/subcommand:service")
		return
	}
	commandOption := args[0]
	switch commandOption {
	case "init":
		if len(args) < 2 {
			base.Fatalf("service init requires a service name")
		}
		serviceName := args[1]
		base.CreateServiceScaffold(serviceName)
		counter.Inc("go/subcommand:service-init")
		return
	default:
		base.Fatalf("unknown service subcommand: %s", commandOption)
	}
}
