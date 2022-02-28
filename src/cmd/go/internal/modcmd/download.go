// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modcmd

import (
	"context"
	"encoding/json"
	"os"
	"runtime"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/modfetch"
	"cmd/go/internal/modload"

	"golang.org/x/mod/module"
	"golang.org/x/mod/semver"
)

var cmdDownload = &base.Command{
	UsageLine: "go mod download [-x] [-json] [modules]",
	Short:     "download modules to local cache",
	Long: `
Download downloads the named modules, which can be module patterns selecting
dependencies of the main module or module queries of the form path@version.

With no arguments, download applies to the modules needed to build and test
the packages in the main module: the modules explicitly required by the main
module if it is at 'go 1.17' or higher, or all transitively-required modules
if at 'go 1.16' or lower.

The go command will automatically download modules as needed during ordinary
execution. The "go mod download" command is useful mainly for pre-filling
the local cache or to compute the answers for a Go module proxy.

By default, download writes nothing to standard output. It may print progress
messages and errors to standard error.

The -json flag causes download to print a sequence of JSON objects
to standard output, describing each downloaded module (or failure),
corresponding to this Go struct:

    type Module struct {
        Path     string // module path
        Version  string // module version
        Error    string // error loading module
        Info     string // absolute path to cached .info file
        GoMod    string // absolute path to cached .mod file
        Zip      string // absolute path to cached .zip file
        Dir      string // absolute path to cached source root directory
        Sum      string // checksum for path, version (as in go.sum)
        GoModSum string // checksum for go.mod (as in go.sum)
    }

The -x flag causes download to print the commands download executes.

See https://golang.org/ref/mod#go-mod-download for more about 'go mod download'.

See https://golang.org/ref/mod#version-queries for more about version queries.
	`,
}

var downloadJSON = cmdDownload.Flag.Bool("json", false, "")

func init() {
	cmdDownload.Run = runDownload // break init cycle

	// TODO(jayconrod): https://golang.org/issue/35849 Apply -x to other 'go mod' commands.
	cmdDownload.Flag.BoolVar(&cfg.BuildX, "x", false, "")
	base.AddModCommonFlags(&cmdDownload.Flag)
}

type moduleJSON struct {
	Path     string `json:",omitempty"`
	Version  string `json:",omitempty"`
	Error    string `json:",omitempty"`
	Info     string `json:",omitempty"`
	GoMod    string `json:",omitempty"`
	Zip      string `json:",omitempty"`
	Dir      string `json:",omitempty"`
	Sum      string `json:",omitempty"`
	GoModSum string `json:",omitempty"`
}

func runDownload(ctx context.Context, cmd *base.Command, args []string) {
	modload.InitWorkfile()

	// Check whether modules are enabled and whether we're in a module.
	modload.ForceUseModules = true
	modload.ExplicitWriteGoMod = true
	haveExplicitArgs := len(args) > 0

	if modload.HasModRoot() || modload.WorkFilePath() != "" {
		modload.LoadModFile(ctx) // to fill MainModules

		if haveExplicitArgs {
			for _, mainModule := range modload.MainModules.Versions() {
				targetAtUpgrade := mainModule.Path + "@upgrade"
				targetAtPatch := mainModule.Path + "@patch"
				for _, arg := range args {
					switch arg {
					case mainModule.Path, targetAtUpgrade, targetAtPatch:
						os.Stderr.WriteString("go: skipping download of " + arg + " that resolves to the main module\n")
					}
				}
			}
		} else if modload.WorkFilePath() != "" {
			// TODO(#44435): Think about what the correct query is to download the
			// right set of modules. Also see code review comment at
			// https://go-review.googlesource.com/c/go/+/359794/comments/ce946a80_6cf53992.
			args = []string{"all"}
		} else {
			mainModule := modload.MainModules.Versions()[0]
			modFile := modload.MainModules.ModFile(mainModule)
			if modFile.Go == nil || semver.Compare("v"+modFile.Go.Version, modload.ExplicitIndirectVersionV) < 0 {
				if len(modFile.Require) > 0 {
					args = []string{"all"}
				}
			} else {
				// As of Go 1.17, the go.mod file explicitly requires every module
				// that provides any package imported by the main module.
				// 'go mod download' is typically run before testing packages in the
				// main module, so by default we shouldn't download the others
				// (which are presumed irrelevant to the packages in the main module).
				// See https://golang.org/issue/44435.
				//
				// However, we also need to load the full module graph, to ensure that
				// we have downloaded enough of the module graph to run 'go list all',
				// 'go mod graph', and similar commands.
				_ = modload.LoadModGraph(ctx, "")

				for _, m := range modFile.Require {
					args = append(args, m.Mod.Path)
				}
			}
		}
	}

	if len(args) == 0 {
		if modload.HasModRoot() {
			os.Stderr.WriteString("go: no module dependencies to download\n")
		} else {
			base.Errorf("go: no modules specified (see 'go help mod download')")
		}
		base.Exit()
	}

	downloadModule := func(m *moduleJSON) {
		var err error
		m.Info, err = modfetch.InfoFile(m.Path, m.Version)
		if err != nil {
			m.Error = err.Error()
			return
		}
		m.GoMod, err = modfetch.GoModFile(m.Path, m.Version)
		if err != nil {
			m.Error = err.Error()
			return
		}
		m.GoModSum, err = modfetch.GoModSum(m.Path, m.Version)
		if err != nil {
			m.Error = err.Error()
			return
		}
		mod := module.Version{Path: m.Path, Version: m.Version}
		m.Zip, err = modfetch.DownloadZip(ctx, mod)
		if err != nil {
			m.Error = err.Error()
			return
		}
		m.Sum = modfetch.Sum(mod)
		m.Dir, err = modfetch.Download(ctx, mod)
		if err != nil {
			m.Error = err.Error()
			return
		}
	}

	var mods []*moduleJSON
	type token struct{}
	sem := make(chan token, runtime.GOMAXPROCS(0))
	infos, infosErr := modload.ListModules(ctx, args, 0)
	if !haveExplicitArgs {
		// 'go mod download' is sometimes run without arguments to pre-populate the
		// module cache. It may fetch modules that aren't needed to build packages
		// in the main module. This is usually not intended, so don't save sums for
		// downloaded modules (golang.org/issue/45332). We do still fix
		// inconsistencies in go.mod though.
		//
		// TODO(#45551): In the future, report an error if go.mod or go.sum need to
		// be updated after loading the build list. This may require setting
		// the mode to "mod" or "readonly" depending on haveExplicitArgs.
		if err := modload.WriteGoMod(ctx); err != nil {
			base.Fatalf("go: %v", err)
		}
	}

	for _, info := range infos {
		if info.Replace != nil {
			info = info.Replace
		}
		if info.Version == "" && info.Error == nil {
			// main module or module replaced with file path.
			// Nothing to download.
			continue
		}
		m := &moduleJSON{
			Path:    info.Path,
			Version: info.Version,
		}
		mods = append(mods, m)
		if info.Error != nil {
			m.Error = info.Error.Err
			continue
		}
		sem <- token{}
		go func() {
			downloadModule(m)
			<-sem
		}()
	}

	// Fill semaphore channel to wait for goroutines to finish.
	for n := cap(sem); n > 0; n-- {
		sem <- token{}
	}

	if *downloadJSON {
		for _, m := range mods {
			b, err := json.MarshalIndent(m, "", "\t")
			if err != nil {
				base.Fatalf("go: %v", err)
			}
			os.Stdout.Write(append(b, '\n'))
			if m.Error != "" {
				base.SetExitStatus(1)
			}
		}
	} else {
		for _, m := range mods {
			if m.Error != "" {
				base.Errorf("go: %v", m.Error)
			}
		}
		base.ExitIfErrors()
	}

	// If there were explicit arguments, update go.mod and especially go.sum.
	// 'go mod download mod@version' is a useful way to add a sum without using
	// 'go get mod@version', which may have other side effects. We print this in
	// some error message hints.
	//
	// Don't save sums for 'go mod download' without arguments; see comment above.
	if haveExplicitArgs {
		if err := modload.WriteGoMod(ctx); err != nil {
			base.Errorf("go: %v", err)
		}
	}

	// If there was an error matching some of the requested packages, emit it now
	// (after we've written the checksums for the modules that were downloaded
	// successfully).
	if infosErr != nil {
		base.Errorf("go: %v", infosErr)
	}
}
