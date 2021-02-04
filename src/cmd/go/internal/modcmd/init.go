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
	UsageLine: "go mod init [module]",
	Short:     "initialize new module in current directory",
	Long: `
Init initializes and writes a new go.mod file in the current directory, in
effect creating a new module rooted at the current directory. The go.mod file
must not already exist.

Init accepts one optional argument, the module path for the new module. If the
module path argument is omitted, init will attempt to infer the module path
using import comments in .go files, vendoring tool configuration files (like
Gopkg.lock), and the current directory (if in GOPATH).

If a configuration file for a vendoring tool is present, init will attempt to
import module requirements from it.

See https://golang.org/ref/mod#go-mod-init for more about 'go mod init'.
`,
	Run: runInit,
}

func init() {
	base.AddModCommonFlags(&cmdInit.Flag)
}

func runInit(ctx context.Context, cmd *base.Command, args []string) {
	if len(args) > 1 {
		base.Fatalf("go mod init: too many arguments")
	}
	var modPath string
	if len(args) == 1 {
		modPath = args[0]
	}

	modload.ForceUseModules = true
	modload.CreateModFile(ctx, modPath) // does all the hard work
}
