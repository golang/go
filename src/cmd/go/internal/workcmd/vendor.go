// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package workcmd

import (
	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/modcmd"
	"cmd/go/internal/modload"
	"context"
)

var cmdVendor = &base.Command{
	UsageLine: "go work vendor [-e] [-v] [-o outdir]",
	Short:     "make vendored copy of dependencies",
	Long: `
Vendor resets the workspace's vendor directory to include all packages
needed to build and test all the workspace's packages.
It does not include test code for vendored packages.

The -v flag causes vendor to print the names of vendored
modules and packages to standard error.

The -e flag causes vendor to attempt to proceed despite errors
encountered while loading packages.

The -o flag causes vendor to create the vendor directory at the given
path instead of "vendor". The go command can only use a vendor directory
named "vendor" within the module root directory, so this flag is
primarily useful for other tools.`,

	Run: runVendor,
}

var vendorE bool   // if true, report errors but proceed anyway
var vendorO string // if set, overrides the default output directory

func init() {
	cmdVendor.Flag.BoolVar(&cfg.BuildV, "v", false, "")
	cmdVendor.Flag.BoolVar(&vendorE, "e", false, "")
	cmdVendor.Flag.StringVar(&vendorO, "o", "", "")
	base.AddChdirFlag(&cmdVendor.Flag)
	base.AddModCommonFlags(&cmdVendor.Flag)
}

func runVendor(ctx context.Context, cmd *base.Command, args []string) {
	modload.InitWorkfile(modload.LoaderState)
	if modload.WorkFilePath(modload.LoaderState) == "" {
		base.Fatalf("go: no go.work file found\n\t(run 'go work init' first or specify path using GOWORK environment variable)")
	}

	modcmd.RunVendor(modload.LoaderState, ctx, vendorE, vendorO, args)
}
