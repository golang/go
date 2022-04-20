// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package modcmd implements the “go mod” command.
package modcmd

import (
	"cmd/go/internal/base"
)

var CmdMod = &base.Command{
	UsageLine: "go mod",
	Short:     "module maintenance",
	Long: `Go mod provides access to operations on modules.

Note that support for modules is built into all the go commands,
not just 'go mod'. For example, day-to-day adding, removing, upgrading,
and downgrading of dependencies should be done using 'go get'.
See 'go help modules' for an overview of module functionality.
	`,

	Commands: []*base.Command{
		cmdDownload,
		cmdEdit,
		cmdGraph,
		cmdInit,
		cmdTidy,
		cmdVendor,
		cmdVerify,
		cmdWhy,
	},
}
