// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// go mod init

package modcmd

import (
	"cmd/go/internal/base"
	"cmd/go/internal/modload"
	"context"
)

var cmdInit = &base.Command{
	UsageLine: "go mod init [module-path]",
	Short:     "initialize new module in current directory",
	Long: `
Init initializes and writes a new go.mod file in the current directory, in
effect creating a new module rooted at the current directory. The go.mod file
must not already exist.

Init accepts one optional argument, the module path for the new module. If the
module path argument is omitted, init will attempt to infer the module path
using import comments in .go files and the current directory (if in GOPATH).

The go directive in the new go.mod file will be set to one minor version
below the current toolchain version (for example, go 1.25.0 when using
the Go 1.26 toolchain). Use 'go get go@version' to change the go version
after initialization.

See https://go.dev/ref/mod#go-mod-init for more about 'go mod init'.
`,
	Run: runInit,
}

func init() {
	base.AddChdirFlag(&cmdInit.Flag)
	base.AddModCommonFlags(&cmdInit.Flag)
}

func runInit(ctx context.Context, cmd *base.Command, args []string) {
	moduleLoader := modload.NewLoader()
	if len(args) > 1 {
		base.Fatalf("go: 'go mod init' accepts at most one argument")
	}
	var modPath string
	if len(args) == 1 {
		modPath = args[0]
	}

	moduleLoader.ForceUseModules = true
	modload.CreateModFile(moduleLoader, ctx, modPath) // does all the hard work
}
