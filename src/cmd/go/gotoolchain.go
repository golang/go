// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !js && !wasip1

package main

import (
	"context"
	"fmt"
	"go/build"
	"internal/godebug"
	"io/fs"
	"log"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"syscall"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/gover"
	"cmd/go/internal/modcmd"
	"cmd/go/internal/modfetch"
	"cmd/go/internal/modload"
	"cmd/go/internal/run"

	"golang.org/x/mod/module"
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
		// cfg.Getenv should fall back to $GOROOT/go.env,
		// so this should not happen, unless a packager
		// has deleted the GOTOOLCHAIN line from go.env.
		// It can also happen if GOROOT is missing or broken,
		// in which case best to let the go command keep running
		// and diagnose the problem.
		return
	}

	var minToolchain, minVers string
	if x, y, ok := strings.Cut(gotoolchain, "+"); ok { // go1.2.3+auto
		orig := gotoolchain
		minToolchain, gotoolchain = x, y
		minVers = gover.ToolchainVersion(minToolchain)
		if minVers == "" {
			base.Fatalf("invalid GOTOOLCHAIN %q: invalid minimum toolchain %q", orig, minToolchain)
		}
		if gotoolchain != "auto" && gotoolchain != "path" {
			base.Fatalf("invalid GOTOOLCHAIN %q: only version suffixes are +auto and +path", orig)
		}
	} else {
		minVers = gover.Local()
		minToolchain = "go" + minVers
	}

	pathOnly := gotoolchain == "path"
	if gotoolchain == "auto" || gotoolchain == "path" {
		gotoolchain = minToolchain

		// Locate and read go.mod or go.work.
		// For go install m@v, it's the installed module's go.mod.
		if m, goVers, ok := goInstallVersion(); ok {
			if gover.Compare(goVers, minVers) > 0 {
				// Always print, because otherwise there's no way for the user to know
				// that a non-default toolchain version is being used here.
				// (Normally you can run "go version", but go install m@v ignores the
				// context that "go version" works in.)
				fmt.Fprintf(os.Stderr, "go: using go%s for %v\n", goVers, m)
				gotoolchain = "go" + goVers
			}
		} else {
			goVers, toolchain := modGoToolchain()
			if toolchain == "local" {
				// Local means always use the default local toolchain,
				// which is already set, so nothing to do here.
				// Note that if we have Go 1.21 installed originally,
				// GOTOOLCHAIN=go1.30.0+auto or GOTOOLCHAIN=go1.30.0,
				// and the go.mod  says "toolchain local", we use Go 1.30, not Go 1.21.
				// That is, local overrides the "auto" part of the calculation
				// but not the minimum that the user has set.
				// Of course, if the go.mod also says "go 1.35", using Go 1.30
				// will provoke an error about the toolchain being too old.
				// That's what people who use toolchain local want:
				// only ever use the toolchain configured in the local system
				// (including its environment and go env -w file).
			} else if toolchain != "" {
				// Accept toolchain only if it is >= our min.
				toolVers := gover.ToolchainVersion(toolchain)
				if gover.Compare(toolVers, minVers) > 0 {
					gotoolchain = toolchain
				}
			} else {
				if gover.Compare(goVers, minVers) > 0 {
					gotoolchain = "go" + goVers
				}
			}
		}
	}

	if gotoolchain == "local" || gotoolchain == "go"+gover.Local() {
		// Let the current binary handle the command.
		return
	}

	// Minimal sanity check of GOTOOLCHAIN setting before search.
	// We want to allow things like go1.20.3 but also gccgo-go1.20.3.
	// We want to disallow mistakes / bad ideas like GOTOOLCHAIN=bash,
	// since we will find that in the path lookup.
	// gover.ToolchainVersion has already done this check (except for the 1)
	// but doing it again makes sure we don't miss it on unexpected code paths.
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
	if pathOnly {
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

// modGoToolchain finds the enclosing go.work or go.mod file
// and returns the go version and toolchain lines from the file.
// The toolchain line overrides the version line
func modGoToolchain() (goVers, toolchain string) {
	wd := base.UncachedCwd()
	file := modload.FindGoWork(wd)
	// $GOWORK can be set to a file that does not yet exist, if we are running 'go work init'.
	// Do not try to load the file in that case
	if _, err := os.Stat(file); err != nil {
		file = ""
	}
	if file == "" {
		file = modload.FindGoMod(wd)
	}
	if file == "" {
		return "", ""
	}

	data, err := os.ReadFile(file)
	if err != nil {
		base.Fatalf("%v", err)
	}
	return gover.GoModLookup(data, "go"), gover.GoModLookup(data, "toolchain")
}

// goInstallVersion looks at the command line to see if it is go install m@v or go run m@v.
// If so, it returns the m@v and the go version from that module's go.mod.
func goInstallVersion() (m module.Version, goVers string, ok bool) {
	// Note: We assume there are no flags between 'go' and 'install' or 'run'.
	// During testing there are some debugging flags that are accepted
	// in that position, but in production go binaries there are not.
	if len(os.Args) < 3 || (os.Args[1] != "install" && os.Args[1] != "run") {
		return module.Version{}, "", false
	}

	var arg string
	switch os.Args[1] {
	case "install":
		// Cannot parse 'go install' command line precisely, because there
		// may be new flags we don't know about. Instead, assume the final
		// argument is a pkg@version we can use.
		arg = os.Args[len(os.Args)-1]
	case "run":
		// For run, the pkg@version can be anywhere on the command line.
		// We don't know the flags, so we can't strictly speaking do this correctly.
		// We do the best we can by interrogating the CmdRun flags and assume
		// that any unknown flag does not take an argument.
		args := os.Args[2:]
		for i := 0; i < len(args); i++ {
			a := args[i]
			if !strings.HasPrefix(a, "-") {
				arg = a
				break
			}
			if a == "-" {
				break
			}
			if a == "--" {
				if i+1 < len(args) {
					arg = args[i+1]
				}
				break
			}
			a = strings.TrimPrefix(a, "-")
			a = strings.TrimPrefix(a, "-")
			if strings.HasPrefix(a, "-") {
				// non-flag but also non-m@v
				break
			}
			if strings.Contains(a, "=") {
				// already has value
				continue
			}
			f := run.CmdRun.Flag.Lookup(a)
			if f == nil {
				// Unknown flag. Assume it doesn't take a value: best we can do.
				continue
			}
			if bf, ok := f.Value.(interface{ IsBoolFlag() bool }); ok && bf.IsBoolFlag() {
				// Does not take value.
				continue
			}
			i++ // Does take a value; skip it.
		}
	}
	if !strings.Contains(arg, "@") || build.IsLocalImport(arg) || filepath.IsAbs(arg) {
		return module.Version{}, "", false
	}
	m.Path, m.Version, _ = strings.Cut(arg, "@")
	if m.Path == "" || m.Version == "" || gover.IsToolchain(m.Path) {
		return module.Version{}, "", false
	}

	// We need to resolve the pkg to a module, to find its go.mod.
	// Normally we use the module loading code to grab the full
	// module file tree for pkg and all its path prefixes, checking each
	// for a file tree that contains source code for pkg.
	// We can't do that here, because the modules may use newer versions
	// of Go that affect which files are contained in the modules and therefore
	// affect their checksums: there is no guarantee an older version of Go
	// can extract a newer Go module from a VCS repo and choose the right files
	// (this allows evolution such as https://go.dev/issue/42965).
	// Instead, we check for a module at all path prefixes (including path itself)
	// and take the max of the Go versions along the path.
	var paths []string
	for len(m.Path) > 1 {
		paths = append(paths, m.Path)
		m.Path = path.Dir(m.Path)
	}
	goVersions := make([]string, len(paths))
	var wg sync.WaitGroup
	for i, path := range paths {
		i := i
		path := path
		wg.Add(1)
		go func() {
			defer wg.Done()
			// TODO(rsc): m.Version could in general be something like latest or patch or upgrade.
			// Use modload.Query. See review comment on https://go.dev/cl/497079.
			data, err := modfetch.GoMod(context.Background(), path, m.Version)
			if err != nil {
				return
			}
			goVersions[i] = gover.GoModLookup(data, "go")
		}()
	}
	wg.Wait()
	goVers = ""
	for i, v := range goVersions {
		if gover.Compare(goVers, v) < 0 {
			m.Path = paths[i]
			goVers = v
		}
	}
	return m, goVers, true
}
