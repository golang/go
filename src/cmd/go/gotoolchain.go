// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !js

package main

import (
	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/modcmd"
	"cmd/go/internal/modload"
	"context"
	"fmt"
	"internal/godebug"
	"io/fs"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
)

const (
	// We download golang.org/toolchain version v0.0.1-<gotoolchain>.<goos>-<goarch>.
	// If the 0.0.1 indicates anything at all, its the version of the toolchain packaging:
	// if for some reason we needed to change the way toolchains are packaged into
	// module zip files in a future version of Go, we could switch to v0.0.2 and then
	// older versions expecting the old format could use v0.0.1 and newer versions
	// would use v0.0.2. Of course, then we'd also have to publish two of each
	// module zip file. It's not likely we'll ever need to change this.
	gotoolchainModule  = "golang.org/toolchain"
	gotoolchainVersion = "v0.0.1"

	// gotoolchainSwitchEnv is a special environment variable
	// set to 1 during the toolchain switch by the parent process
	// and cleared in the child process. When set, that indicates
	// to the child not to do its own toolchain switch logic,
	// to avoid an infinite recursion if for some reason a toolchain
	// did not believe it could handle its own version and then
	// reinvoked itself.
	gotoolchainSwitchEnv = "GOTOOLCHAIN_INTERNAL_SWITCH"
)

// switchGoToolchain invokes a different Go toolchain if directed by
// the GOTOOLCHAIN environment variable or the user's configuration
// or go.mod file.
func switchGoToolchain() {
	log.SetPrefix("go: ")
	defer log.SetPrefix("")

	sw := os.Getenv(gotoolchainSwitchEnv)
	os.Unsetenv(gotoolchainSwitchEnv)

	if !modload.WillBeEnabled() || sw == "1" {
		return
	}

	gotoolchain := cfg.Getenv("GOTOOLCHAIN")
	if gotoolchain == "" {
		if strings.HasPrefix(runtime.Version(), "go") {
			gotoolchain = "local" // TODO: set to "auto" once auto is implemented below
		} else {
			gotoolchain = "local"
		}
	}
	env := gotoolchain
	if gotoolchain == "auto" || gotoolchain == "path" {
		// TODO: Locate and read go.mod or go.work.
		base.Fatalf("GOTOOLCHAIN=auto not yet implemented")
	}

	if gotoolchain == "local" || gotoolchain == runtime.Version() {
		// Let the current binary handle the command.
		return
	}

	// Minimal sanity check of GOTOOLCHAIN setting before search.
	// We want to allow things like go1.20.3 but also gccgo-go1.20.3.
	// We want to disallow mistakes / bad ideas like GOTOOLCHAIN=bash,
	// since we will find that in the path lookup.
	if !strings.HasPrefix(gotoolchain, "go1") && !strings.Contains(gotoolchain, "-go1") {
		base.Fatalf("invalid GOTOOLCHAIN %q", gotoolchain)
	}

	// Look in PATH for the toolchain before we download one.
	// This allows custom toolchains as well as reuse of toolchains
	// already installed using go install golang.org/dl/go1.2.3@latest.
	if exe, err := exec.LookPath(gotoolchain); err == nil {
		execGoToolchain(gotoolchain, "", exe)
	}

	// GOTOOLCHAIN=auto looks in PATH and then falls back to download.
	// GOTOOLCHAIN=path only looks in PATH.
	if env == "path" {
		base.Fatalf("cannot find %q in PATH", gotoolchain)
	}

	// Set up modules without an explicit go.mod, to download distribution.
	modload.ForceUseModules = true
	modload.RootMode = modload.NoRoot
	modload.Init()

	// Download and unpack toolchain module into module cache.
	// Note that multiple go commands might be doing this at the same time,
	// and that's OK: the module cache handles that case correctly.
	m := &modcmd.ModuleJSON{
		Path:    gotoolchainModule,
		Version: gotoolchainVersion + "-" + gotoolchain + "." + runtime.GOOS + "-" + runtime.GOARCH,
	}
	modcmd.DownloadModule(context.Background(), m)
	if m.Error != "" {
		if strings.Contains(m.Error, ".info: 404") {
			base.Fatalf("download %s for %s/%s: toolchain not available", gotoolchain, runtime.GOOS, runtime.GOARCH)
		}
		base.Fatalf("download %s: %v", gotoolchain, m.Error)
	}

	// On first use after download, set the execute bits on the commands
	// so that we can run them. Note that multiple go commands might be
	// doing this at the same time, but if so no harm done.
	dir := m.Dir
	if runtime.GOOS != "windows" {
		info, err := os.Stat(filepath.Join(dir, "bin/go"))
		if err != nil {
			base.Fatalf("download %s: %v", gotoolchain, err)
		}
		if info.Mode()&0111 == 0 {
			// allowExec sets the exec permission bits on all files found in dir.
			allowExec := func(dir string) {
				err := filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
					if err != nil {
						return err
					}
					if !d.IsDir() {
						info, err := os.Stat(path)
						if err != nil {
							return err
						}
						if err := os.Chmod(path, info.Mode()&0777|0111); err != nil {
							return err
						}
					}
					return nil
				})
				if err != nil {
					base.Fatalf("download %s: %v", gotoolchain, err)
				}
			}

			// Set the bits in pkg/tool before bin/go.
			// If we are racing with another go command and do bin/go first,
			// then the check of bin/go above might succeed, the other go command
			// would skip its own mode-setting, and then the go command might
			// try to run a tool before we get to setting the bits on pkg/tool.
			// Setting pkg/tool before bin/go avoids that ordering problem.
			// The only other tool the go command invokes is gofmt,
			// so we set that one explicitly before handling bin (which will include bin/go).
			allowExec(filepath.Join(dir, "pkg/tool"))
			allowExec(filepath.Join(dir, "bin/gofmt"))
			allowExec(filepath.Join(dir, "bin"))
		}
	}

	// Reinvoke the go command.
	execGoToolchain(gotoolchain, dir, filepath.Join(dir, "bin/go"))
}

// execGoToolchain execs the Go toolchain with the given name (gotoolchain),
// GOROOT directory, and go command executable.
// The GOROOT directory is empty if we are invoking a command named
// gotoolchain found in $PATH.
func execGoToolchain(gotoolchain, dir, exe string) {
	os.Setenv(gotoolchainSwitchEnv, "1")
	if dir == "" {
		os.Unsetenv("GOROOT")
	} else {
		os.Setenv("GOROOT", dir)
	}

	// On Windows, there is no syscall.Exec, so the best we can do
	// is run a subprocess and exit with the same status.
	// Doing the same on Unix would be a problem because it wouldn't
	// propagate signals and such, but there are no signals on Windows.
	// We also use the exec case when GODEBUG=gotoolchainexec=0,
	// to allow testing this code even when not on Windows.
	if godebug.New("#gotoolchainexec").Value() == "0" || runtime.GOOS == "windows" {
		cmd := exec.Command(exe, os.Args[1:]...)
		if runtime.GOOS == "windows" && strings.Contains(exe, "go1.999test") {
			// See testdata/script/gotoolchain.txt.
			cmd = exec.Command("cmd", "/c", "echo pretend we ran "+exe)
		}
		cmd.Stdin = os.Stdin
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		fmt.Fprintln(os.Stderr, cmd.Args)
		err := cmd.Run()
		if err != nil {
			if e, ok := err.(*exec.ExitError); ok && e.ProcessState != nil {
				if e.ProcessState.Exited() {
					os.Exit(e.ProcessState.ExitCode())
				}
				base.Fatalf("exec %s: %s", gotoolchain, e.ProcessState)
			}
			base.Fatalf("exec %s: %s", exe, err)
		}
		os.Exit(0)
	}
	err := syscall.Exec(exe, os.Args, os.Environ())
	base.Fatalf("exec %s: %v", gotoolchain, err)
}
