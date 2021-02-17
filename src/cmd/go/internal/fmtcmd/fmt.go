// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package fmtcmd implements the ``go fmt'' command.
package fmtcmd

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sync"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/load"
	"cmd/go/internal/modload"
	"cmd/go/internal/str"
)

func init() {
	base.AddBuildFlagsNX(&CmdFmt.Flag)
	base.AddModFlag(&CmdFmt.Flag)
	base.AddModCommonFlags(&CmdFmt.Flag)
}

var CmdFmt = &base.Command{
	Run:       runFmt,
	UsageLine: "go fmt [-n] [-x] [packages]",
	Short:     "gofmt (reformat) package sources",
	Long: `
Fmt runs the command 'gofmt -l -w' on the packages named
by the import paths. It prints the names of the files that are modified.

For more about gofmt, see 'go doc cmd/gofmt'.
For more about specifying packages, see 'go help packages'.

The -n flag prints commands that would be executed.
The -x flag prints commands as they are executed.

The -mod flag's value sets which module download mode
to use: readonly or vendor. See 'go help modules' for more.

To run gofmt with specific options, run gofmt itself.

See also: go fix, go vet.
	`,
}

func runFmt(ctx context.Context, cmd *base.Command, args []string) {
	printed := false
	gofmt := gofmtPath()
	procs := runtime.GOMAXPROCS(0)
	var wg sync.WaitGroup
	wg.Add(procs)
	fileC := make(chan string, 2*procs)
	for i := 0; i < procs; i++ {
		go func() {
			defer wg.Done()
			for file := range fileC {
				base.Run(str.StringList(gofmt, "-l", "-w", file))
			}
		}()
	}
	for _, pkg := range load.PackagesAndErrors(ctx, args) {
		if modload.Enabled() && pkg.Module != nil && !pkg.Module.Main {
			if !printed {
				fmt.Fprintf(os.Stderr, "go: not formatting packages in dependency modules\n")
				printed = true
			}
			continue
		}
		if pkg.Error != nil {
			var nogo *load.NoGoError
			var embed *load.EmbedError
			if (errors.As(pkg.Error, &nogo) || errors.As(pkg.Error, &embed)) && len(pkg.InternalAllGoFiles()) > 0 {
				// Skip this error, as we will format
				// all files regardless.
			} else {
				base.Errorf("%v", pkg.Error)
				continue
			}
		}
		// Use pkg.gofiles instead of pkg.Dir so that
		// the command only applies to this package,
		// not to packages in subdirectories.
		files := base.RelPaths(pkg.InternalAllGoFiles())
		for _, file := range files {
			fileC <- file
		}
	}
	close(fileC)
	wg.Wait()
}

func gofmtPath() string {
	gofmt := "gofmt"
	if base.ToolIsWindows {
		gofmt += base.ToolWindowsExtension
	}

	gofmtPath := filepath.Join(cfg.GOBIN, gofmt)
	if _, err := os.Stat(gofmtPath); err == nil {
		return gofmtPath
	}

	gofmtPath = filepath.Join(cfg.GOROOT, "bin", gofmt)
	if _, err := os.Stat(gofmtPath); err == nil {
		return gofmtPath
	}

	// fallback to looking for gofmt in $PATH
	return "gofmt"
}
