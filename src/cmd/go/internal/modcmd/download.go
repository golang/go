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
)

var cmdDownload = &base.Command{
	UsageLine: "go mod download [-x] [-json] [modules]",
	Short:     "download modules to local cache",
	Long: `
Download downloads the named modules, which can be module patterns selecting
dependencies of the main module or module queries of the form path@version.
With no arguments, download applies to all dependencies of the main module
(equivalent to 'go mod download all').

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
	// Check whether modules are enabled and whether we're in a module.
	modload.ForceUseModules = true
	if !modload.HasModRoot() && len(args) == 0 {
		base.Fatalf("go mod download: no modules specified (see 'go help mod download')")
	}
	haveExplicitArgs := len(args) > 0
	if !haveExplicitArgs {
		args = []string{"all"}
	}
	if modload.HasModRoot() {
		modload.LoadModFile(ctx) // to fill Target
		targetAtUpgrade := modload.Target.Path + "@upgrade"
		targetAtPatch := modload.Target.Path + "@patch"
		for _, arg := range args {
			switch arg {
			case modload.Target.Path, targetAtUpgrade, targetAtPatch:
				os.Stderr.WriteString("go mod download: skipping argument " + arg + " that resolves to the main module\n")
			}
		}
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
	listU := false
	listVersions := false
	listRetractions := false
	type token struct{}
	sem := make(chan token, runtime.GOMAXPROCS(0))
	infos := modload.ListModules(ctx, args, listU, listVersions, listRetractions)
	if !haveExplicitArgs {
		// 'go mod download' is sometimes run without arguments to pre-populate
		// the module cache. It may fetch modules that aren't needed to build
		// packages in the main mdoule. This is usually not intended, so don't save
		// sums for downloaded modules (golang.org/issue/45332).
		// TODO(golang.org/issue/45551): For now, save sums needed to load the
		// build list (same as 1.15 behavior). In the future, report an error if
		// go.mod or go.sum need to be updated after loading the build list.
		modload.WriteGoMod()
		modload.DisallowWriteGoMod()
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
				base.Fatalf("go mod download: %v", err)
			}
			os.Stdout.Write(append(b, '\n'))
			if m.Error != "" {
				base.SetExitStatus(1)
			}
		}
	} else {
		for _, m := range mods {
			if m.Error != "" {
				base.Errorf("go mod download: %v", m.Error)
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
		modload.WriteGoMod()
	}
}
