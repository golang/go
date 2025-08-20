// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modcmd

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"runtime"
	"sync"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/gover"
	"cmd/go/internal/modfetch"
	"cmd/go/internal/modfetch/codehost"
	"cmd/go/internal/modload"
	"cmd/go/internal/toolchain"

	"golang.org/x/mod/module"
)

var cmdDownload = &base.Command{
	UsageLine: "go mod download [-x] [-json] [-reuse=old.json] [modules]",
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
        Query    string // version query corresponding to this version
        Version  string // module version
        Error    string // error loading module
        Info     string // absolute path to cached .info file
        GoMod    string // absolute path to cached .mod file
        Zip      string // absolute path to cached .zip file
        Dir      string // absolute path to cached source root directory
        Sum      string // checksum for path, version (as in go.sum)
        GoModSum string // checksum for go.mod (as in go.sum)
        Origin   any    // provenance of module
        Reuse    bool   // reuse of old module info is safe
    }

The -reuse flag accepts the name of file containing the JSON output of a
previous 'go mod download -json' invocation. The go command may use this
file to determine that a module is unchanged since the previous invocation
and avoid redownloading it. Modules that are not redownloaded will be marked
in the new output by setting the Reuse field to true. Normally the module
cache provides this kind of reuse automatically; the -reuse flag can be
useful on systems that do not preserve the module cache.

The -x flag causes download to print the commands download executes.

See https://golang.org/ref/mod#go-mod-download for more about 'go mod download'.

See https://golang.org/ref/mod#version-queries for more about version queries.
	`,
}

var (
	downloadJSON  = cmdDownload.Flag.Bool("json", false, "")
	downloadReuse = cmdDownload.Flag.String("reuse", "", "")
)

func init() {
	cmdDownload.Run = runDownload // break init cycle

	// TODO(jayconrod): https://golang.org/issue/35849 Apply -x to other 'go mod' commands.
	cmdDownload.Flag.BoolVar(&cfg.BuildX, "x", false, "")
	base.AddChdirFlag(&cmdDownload.Flag)
	base.AddModCommonFlags(&cmdDownload.Flag)
}

// A ModuleJSON describes the result of go mod download.
type ModuleJSON struct {
	Path     string `json:",omitempty"`
	Version  string `json:",omitempty"`
	Query    string `json:",omitempty"`
	Error    string `json:",omitempty"`
	Info     string `json:",omitempty"`
	GoMod    string `json:",omitempty"`
	Zip      string `json:",omitempty"`
	Dir      string `json:",omitempty"`
	Sum      string `json:",omitempty"`
	GoModSum string `json:",omitempty"`

	Origin *codehost.Origin `json:",omitempty"`
	Reuse  bool             `json:",omitempty"`
}

func runDownload(ctx context.Context, cmd *base.Command, args []string) {
	modload.InitWorkfile()

	// Check whether modules are enabled and whether we're in a module.
	modload.LoaderState.ForceUseModules = true
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
			if modFile.Go == nil || gover.Compare(modFile.Go.Version, gover.ExplicitIndirectVersion) < 0 {
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
				_, err := modload.LoadModGraph(ctx, "")
				if err != nil {
					// TODO(#64008): call base.Fatalf instead of toolchain.SwitchOrFatal
					// here, since we can only reach this point with an outdated toolchain
					// if the go.mod file is inconsistent.
					toolchain.SwitchOrFatal(ctx, err)
				}

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

	if *downloadReuse != "" && modload.HasModRoot() {
		base.Fatalf("go mod download -reuse cannot be used inside a module")
	}

	var mods []*ModuleJSON
	type token struct{}
	sem := make(chan token, runtime.GOMAXPROCS(0))
	infos, infosErr := modload.ListModules(ctx, args, 0, *downloadReuse)

	// There is a bit of a chicken-and-egg problem here: ideally we need to know
	// which Go version to switch to download the requested modules, but if we
	// haven't downloaded the module's go.mod file yet the GoVersion field of its
	// info struct is not yet populated.
	//
	// We also need to be careful to only print the info for each module once
	// if the -json flag is set.
	//
	// In theory we could go through each module in the list, attempt to download
	// its go.mod file, and record the maximum version (either from the file or
	// from the resulting TooNewError), all before we try the actual full download
	// of each module.
	//
	// For now, we go ahead and try all the downloads and collect the errors, and
	// if any download failed due to a TooNewError, we switch toolchains and try
	// again. Any downloads that already succeeded will still be in cache.
	// That won't give optimal concurrency (we'll do two batches of concurrent
	// downloads instead of all in one batch), and it might add a little overhead
	// to look up the downloads from the first batch in the module cache when
	// we see them again in the second batch. On the other hand, it's way simpler
	// to implement, and not really any more expensive if the user is requesting
	// no explicit arguments (their go.mod file should already list an appropriate
	// toolchain version) or only one module (as is used by the Go Module Proxy).

	if infosErr != nil {
		var sw toolchain.Switcher
		sw.Error(infosErr)
		if sw.NeedSwitch() {
			sw.Switch(ctx)
		}
		// Otherwise, wait to report infosErr after we have downloaded
		// when we can.
	}

	if !haveExplicitArgs && modload.WorkFilePath() == "" {
		// 'go mod download' is sometimes run without arguments to pre-populate the
		// module cache. In modules that aren't at go 1.17 or higher, it may fetch
		// modules that aren't needed to build packages in the main module. This is
		// usually not intended, so don't save sums for downloaded modules
		// (golang.org/issue/45332). We do still fix inconsistencies in go.mod
		// though.
		//
		// TODO(#64008): In the future, report an error if go.mod or go.sum need to
		// be updated after loading the build list. This may require setting
		// the mode to "mod" or "readonly" depending on haveExplicitArgs.
		if err := modload.WriteGoMod(ctx, modload.WriteOpts{}); err != nil {
			base.Fatal(err)
		}
	}

	var downloadErrs sync.Map
	for _, info := range infos {
		if info.Replace != nil {
			info = info.Replace
		}
		if info.Version == "" && info.Error == nil {
			// main module or module replaced with file path.
			// Nothing to download.
			continue
		}
		m := &ModuleJSON{
			Path:    info.Path,
			Version: info.Version,
			Query:   info.Query,
			Reuse:   info.Reuse,
			Origin:  info.Origin,
		}
		mods = append(mods, m)
		if info.Error != nil {
			m.Error = info.Error.Err
			continue
		}
		if m.Reuse {
			continue
		}
		sem <- token{}
		go func() {
			err := DownloadModule(ctx, m)
			if err != nil {
				downloadErrs.Store(m, err)
				m.Error = err.Error()
			}
			<-sem
		}()
	}

	// Fill semaphore channel to wait for goroutines to finish.
	for n := cap(sem); n > 0; n-- {
		sem <- token{}
	}

	// If there were explicit arguments
	// (like 'go mod download golang.org/x/tools@latest'),
	// check whether we need to upgrade the toolchain in order to download them.
	//
	// (If invoked without arguments, we expect the module graph to already
	// be tidy and the go.mod file to declare a 'go' version that satisfies
	// transitive requirements. If that invariant holds, then we should have
	// already upgraded when we loaded the module graph, and should not need
	// an additional check here. See https://go.dev/issue/45551.)
	//
	// We also allow upgrades if in a workspace because in workspace mode
	// with no arguments we download the module pattern "all",
	// which may include dependencies that are normally pruned out
	// of the individual modules in the workspace.
	if haveExplicitArgs || modload.WorkFilePath() != "" {
		var sw toolchain.Switcher
		// Add errors to the Switcher in deterministic order so that they will be
		// logged deterministically.
		for _, m := range mods {
			if erri, ok := downloadErrs.Load(m); ok {
				sw.Error(erri.(error))
			}
		}
		// Only call sw.Switch if it will actually switch.
		// Otherwise, we may want to write the errors as JSON
		// (instead of using base.Error as sw.Switch would),
		// and we may also have other errors to report from the
		// initial infos returned by ListModules.
		if sw.NeedSwitch() {
			sw.Switch(ctx)
		}
	}

	if *downloadJSON {
		for _, m := range mods {
			b, err := json.MarshalIndent(m, "", "\t")
			if err != nil {
				base.Fatal(err)
			}
			os.Stdout.Write(append(b, '\n'))
			if m.Error != "" {
				base.SetExitStatus(1)
			}
		}
	} else {
		for _, m := range mods {
			if m.Error != "" {
				base.Error(errors.New(m.Error))
			}
		}
		base.ExitIfErrors()
	}

	// If there were explicit arguments, update go.mod and especially go.sum.
	// 'go mod download mod@version' is a useful way to add a sum without using
	// 'go get mod@version', which may have other side effects. We print this in
	// some error message hints.
	//
	// If we're in workspace mode, update go.work.sum with checksums for all of
	// the modules we downloaded that aren't already recorded. Since a requirement
	// in one module may upgrade a dependency of another, we can't be sure that
	// the import graph matches the import graph of any given module in isolation,
	// so we may end up needing to load packages from modules that wouldn't
	// otherwise be relevant.
	//
	// TODO(#44435): If we adjust the set of modules downloaded in workspace mode,
	// we may also need to adjust the logic for saving checksums here.
	//
	// Don't save sums for 'go mod download' without arguments unless we're in
	// workspace mode; see comment above.
	if haveExplicitArgs || modload.WorkFilePath() != "" {
		if err := modload.WriteGoMod(ctx, modload.WriteOpts{}); err != nil {
			base.Error(err)
		}
	}

	// If there was an error matching some of the requested packages, emit it now
	// (after we've written the checksums for the modules that were downloaded
	// successfully).
	if infosErr != nil {
		base.Error(infosErr)
	}
}

// DownloadModule runs 'go mod download' for m.Path@m.Version,
// leaving the results (including any error) in m itself.
func DownloadModule(ctx context.Context, m *ModuleJSON) error {
	var err error
	_, file, err := modfetch.InfoFile(ctx, m.Path, m.Version)
	if err != nil {
		return err
	}
	m.Info = file
	m.GoMod, err = modfetch.GoModFile(ctx, m.Path, m.Version)
	if err != nil {
		return err
	}
	m.GoModSum, err = modfetch.GoModSum(ctx, m.Path, m.Version)
	if err != nil {
		return err
	}
	mod := module.Version{Path: m.Path, Version: m.Version}
	m.Zip, err = modfetch.DownloadZip(ctx, mod)
	if err != nil {
		return err
	}
	m.Sum = modfetch.Sum(ctx, mod)
	m.Dir, err = modfetch.Download(ctx, mod)
	return err
}
