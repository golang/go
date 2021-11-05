// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// go mod initwork

package modcmd

import (
	"cmd/go/internal/base"
	"cmd/go/internal/modload"
	"context"
	"path/filepath"
)

var _ = modload.TODOWorkspaces("Add more documentation below. Though this is" +
	"enough for those trying workspaces out, there should be more through" +
	"documentation if the proposal is accepted and released.")

var cmdInitwork = &base.Command{
	UsageLine: "go mod initwork [moddirs]",
	Short:     "initialize workspace file",
	Long: `go mod initwork initializes and writes a new go.work file in the current
directory, in effect creating a new workspace at the current directory.

go mod initwork optionally accepts paths to the workspace modules as arguments.
If the argument is omitted, an empty workspace with no modules will be created.

See the workspaces design proposal at
https://go.googlesource.com/proposal/+/master/design/45713-workspace.md for
more information.
`,
	Run: runInitwork,
}

func init() {
	base.AddModCommonFlags(&cmdInitwork.Flag)
	base.AddWorkfileFlag(&cmdInitwork.Flag)
}

func runInitwork(ctx context.Context, cmd *base.Command, args []string) {
	modload.InitWorkfile()

	modload.ForceUseModules = true

	// TODO(matloob): support using the -workfile path
	// To do that properly, we'll have to make the module directories
	// make dirs relative to workFile path before adding the paths to
	// the directory entries

	workFile := filepath.Join(base.Cwd(), "go.work")

	modload.CreateWorkFile(ctx, workFile, args)
}
