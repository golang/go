// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package workcmd implements the ``go work'' command.
package workcmd

import (
	"cmd/go/internal/base"
)

var CmdWork = &base.Command{
	UsageLine: "go work",
	Short:     "workspace maintenance",
	Long: `Go workspace provides access to operations on worskpaces.

Note that support for workspaces is built into many other commands,
not just 'go work'.

See 'go help modules' for information about Go's module system of
which workspaces are a part.
`,

	Commands: []*base.Command{
		cmdEdit,
		cmdInit,
		cmdSync,
	},
}
