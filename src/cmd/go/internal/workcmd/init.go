// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// go work init

package workcmd

import (
	"cmd/go/internal/base"
	"cmd/go/internal/fsys"
	"cmd/go/internal/gover"
	"cmd/go/internal/modload"
	"context"
	"os"
	"path/filepath"

	"golang.org/x/mod/modfile"
)

var cmdInit = &base.Command{
	UsageLine: "go work init [moddirs]",
	Short:     "initialize workspace file",
	Long: `Init initializes and writes a new go.work file in the
current directory, in effect creating a new workspace at the current
directory.

go work init optionally accepts paths to the workspace modules as
arguments. If the argument is omitted, an empty workspace with no
modules will be created.

Each argument path is added to a use directive in the go.work file. The
current go version will also be listed in the go.work file.

See the workspaces reference at https://go.dev/ref/mod#workspaces
for more information.
`,
	Run: runInit,
}

func init() {
	base.AddChdirFlag(&cmdInit.Flag)
	base.AddModCommonFlags(&cmdInit.Flag)
}

func runInit(ctx context.Context, cmd *base.Command, args []string) {
	modload.InitWorkfile()

	modload.ForceUseModules = true

	workFile := modload.WorkFilePath()
	if workFile == "" {
		workFile = filepath.Join(base.Cwd(), "go.work")
	}

	CreateWorkFile(ctx, workFile, args)
}

// CreateWorkFile initializes a new workspace by creating a go.work file.
func CreateWorkFile(ctx context.Context, workFile string, modDirs []string) {
	if _, err := fsys.Stat(workFile); err == nil {
		base.Fatalf("go: %s already exists", workFile)
	}

	goV := gover.Local() // Use current Go version by default
	wf := new(modfile.WorkFile)
	wf.Syntax = new(modfile.FileSyntax)
	wf.AddGoStmt(goV)

	for _, dir := range modDirs {
		_, f, err := modload.ReadModFile(filepath.Join(dir, "go.mod"), nil)
		if err != nil {
			if os.IsNotExist(err) {
				base.Fatalf("go: creating workspace file: no go.mod file exists in directory %v", dir)
			}
			base.Fatalf("go: error parsing go.mod in directory %s: %v", dir, err)
		}
		wf.AddUse(modload.ToDirectoryPath(dir), f.Module.Mod.Path)
	}

	modload.UpdateWorkFile(wf)
	modload.WriteWorkFile(workFile, wf)
}
