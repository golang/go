// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// go mod tidy

package modcmd

import (
	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/modload"
	"cmd/go/internal/work"
	"context"
)

var cmdTidy = &base.Command{
	UsageLine: "go mod tidy [-v]",
	Short:     "add missing and remove unused modules",
	Long: `
Tidy makes sure go.mod matches the source code in the module.
It adds any missing modules necessary to build the current module's
packages and dependencies, and it removes unused modules that
don't provide any relevant packages. It also adds any missing entries
to go.sum and removes any unnecessary ones.

The -v flag causes tidy to print information about removed modules
to standard error.
	`,
}

func init() {
	cmdTidy.Run = runTidy // break init cycle
	cmdTidy.Flag.BoolVar(&cfg.BuildV, "v", false, "")
	work.AddModCommonFlags(cmdTidy)
}

func runTidy(ctx context.Context, cmd *base.Command, args []string) {
	if len(args) > 0 {
		base.Fatalf("go mod tidy: no arguments allowed")
	}

	modload.LoadALL(ctx)
	modload.TidyBuildList()
	modload.TrimGoSum()
	modload.WriteGoMod()
}
