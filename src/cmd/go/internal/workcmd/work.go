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
	Long: `Go workspace provides access to operations on workspaces.

Note that support for workspaces is built into many other commands,
not just 'go work'.

See 'go help modules' for information about Go's module system of
which workspaces are a part.

A workspace is specified by a go.work file that specifies a set of
module directories with the "use" directive. These modules are used
as root modules by the go command for builds and related operations.
A workspace that does not specify modules to be used cannot be used
to do builds from local modules.

To determine whether the go command is operating in workspace mode,
use the "go env GOWORK" command. This will specify the workspace
file being used.
`,

	Commands: []*base.Command{
		cmdEdit,
		cmdInit,
		cmdSync,
		cmdUse,
	},
}
