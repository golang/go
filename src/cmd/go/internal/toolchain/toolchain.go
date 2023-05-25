// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package toolchain implements dynamic switching of Go toolchains.
package toolchain

import (
	"context"
	"fmt"
	"go/build"
	"io/fs"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"

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

// Switch invokes a different Go toolchain if directed by
// the GOTOOLCHAIN environment variable or the user's configuration
// or go.mod file.
// It must be called early in startup.
func Switch() {
	log.SetPrefix("go: ")
	defer log.SetPrefix("")

	sw := os.Getenv(gotoolchainSwitchEnv)
	os.Unsetenv(gotoolchainSwitchEnv)
	// The sw == "1" check is delayed until later so that we still fill in gover.Startup for use in errors.

	if !modload.WillBeEnabled() {
		return
	}

	gotoolchain := cfg.Getenv("GOTOOLCHAIN")
	gover.Startup.GOTOOLCHAIN = gotoolchain
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
		minVers = gover.FromToolchain(minToolchain)
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
				var err error
				gotoolchain, err = NewerToolchain(context.Background(), goVers)
				if err != nil {
					fmt.Fprintf(os.Stderr, "go: %v\n", err)
					gotoolchain = "go" + goVers
				}
				fmt.Fprintf(os.Stderr, "go: using %s for %v\n", gotoolchain, m)
			}
		} else {
			file, goVers, toolchain := modGoToolchain()
			gover.Startup.AutoFile = file
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
				gover.Startup.AutoToolchain = toolchain
				gotoolchain = "local"
			} else {
				if toolchain != "" {
					// Accept toolchain only if it is >= our min.
					toolVers := gover.FromToolchain(toolchain)
					if toolVers == "" || (!strings.HasPrefix(toolchain, "go") && !strings.Contains(toolchain, "-go")) {
						base.Fatalf("invalid toolchain %q in %s", toolchain, base.ShortPath(file))
					}
					if gover.Compare(toolVers, minVers) >= 0 {
						gotoolchain = toolchain
						minVers = toolVers
						gover.Startup.AutoToolchain = toolchain
					}
				}
				if gover.Compare(goVers, minVers) > 0 {
					gotoolchain = "go" + goVers
					gover.Startup.AutoGoVersion = goVers
					gover.Startup.AutoToolchain = "" // in case we are overriding it for being too old
				}
			}
		}
	}

	if sw == "1" || gotoolchain == "local" || gotoolchain == "go"+gover.Local() {
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

	SwitchTo(gotoolchain)
}

// NewerToolchain returns the name of the toolchain to use when we need
// to reinvoke a newer toolchain that must support at least the given Go version.
//
// If the latest major release is 1.N.0, we use the latest patch release of 1.(N-1) if that's >= version.
// Otherwise we use the latest 1.N if that's allowed.
// Otherwise we use the latest release.
func NewerToolchain(ctx context.Context, version string) (string, error) {
	var versions *modfetch.Versions
	err := modfetch.TryProxies(func(proxy string) error {
		v, err := modfetch.Lookup(ctx, proxy, "go").Versions(ctx, "")
		if err != nil {
			return err
		}
		versions = v
		return nil
	})
	if err != nil {
		return "", err
	}
	return newerToolchain(version, versions.List)
}

// newerToolchain implements NewerToolchain where the list of choices is known.
// It is separated out for easier testing of this logic.
func newerToolchain(need string, list []string) (string, error) {
	// Consider each release in the list, from newest to oldest,
	// considering only entries >= need and then only entries
	// that are the latest in their language family
	// (the latest 1.40, the latest 1.39, and so on).
	// We prefer the latest patch release before the most recent release family,
	// so if the latest release is 1.40.1 we'll take the latest 1.39.X.
	// Failing that, we prefer the latest patch release before the most recent
	// prerelease family, so if the latest release is 1.40rc1 is out but 1.39 is okay,
	// we'll still take 1.39.X.
	// Failing that we'll take the latest release.
	latest := ""
	for i := len(list) - 1; i >= 0; i-- {
		v := list[i]
		if gover.Compare(v, need) < 0 {
			break
		}
		if gover.Lang(latest) == gover.Lang(v) {
			continue
		}
		newer := latest
		latest = v
		if newer != "" && !gover.IsPrerelease(newer) {
			// latest is the last patch release of Go 1.X, and we saw a non-prerelease of Go 1.(X+1),
			// so latest is the one we want.
			break
		}
	}
	if latest == "" {
		return "", fmt.Errorf("no releases found for go >= %v", need)
	}
	return "go" + latest, nil
}

// HasAuto reports whether the GOTOOLCHAIN setting allows "auto" upgrades.
func HasAuto() bool {
	env := cfg.Getenv("GOTOOLCHAIN")
	return env == "auto" || strings.HasSuffix(env, "+auto")
}

// HasPath reports whether the GOTOOLCHAIN setting allows "path" upgrades.
func HasPath() bool {
	env := cfg.Getenv("GOTOOLCHAIN")
	return env == "path" || strings.HasSuffix(env, "+path")
}

// SwitchTo invokes the specified Go toolchain or else prints an error and exits the process.
// If $GOTOOLCHAIN is set to path or min+path, SwitchTo only considers the PATH
// as a source of Go toolchains. Otherwise SwitchTo tries the PATH but then downloads
// a toolchain if necessary.
func SwitchTo(gotoolchain string) {
	log.SetPrefix("go: ")

	env := cfg.Getenv("GOTOOLCHAIN")
	pathOnly := env == "path" || strings.HasSuffix(env, "+path")

	// For testing, if TESTGO_VERSION is already in use
	// (only happens in the cmd/go test binary)
	// and TESTGO_VERSION_SWITCH=1 is set,
	// "switch" toolchains by changing TESTGO_VERSION
	// and reinvoking the current binary.
	if gover.TestVersion != "" && os.Getenv("TESTGO_VERSION_SWITCH") == "1" {
		os.Setenv("TESTGO_VERSION", gotoolchain)
		exe, err := os.Executable()
		if err != nil {
			base.Fatalf("%v", err)
		}
		execGoToolchain(gotoolchain, os.Getenv("GOROOT"), exe)
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
	modload.Reset()
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

// modGoToolchain finds the enclosing go.work or go.mod file
// and returns the go version and toolchain lines from the file.
// The toolchain line overrides the version line
func modGoToolchain() (file, goVers, toolchain string) {
	wd := base.UncachedCwd()
	file = modload.FindGoWork(wd)
	// $GOWORK can be set to a file that does not yet exist, if we are running 'go work init'.
	// Do not try to load the file in that case
	if _, err := os.Stat(file); err != nil {
		file = ""
	}
	if file == "" {
		file = modload.FindGoMod(wd)
	}
	if file == "" {
		return "", "", ""
	}

	data, err := os.ReadFile(file)
	if err != nil {
		base.Fatalf("%v", err)
	}
	return file, gover.GoModLookup(data, "go"), gover.GoModLookup(data, "toolchain")
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

	// Set up modules without an explicit go.mod, to download go.mod.
	modload.ForceUseModules = true
	modload.RootMode = modload.NoRoot
	modload.Init()
	defer modload.Reset()

	// See internal/load.PackagesAndErrorsOutsideModule
	ctx := context.Background()
	allowed := modload.CheckAllowed
	if modload.IsRevisionQuery(m.Path, m.Version) {
		// Don't check for retractions if a specific revision is requested.
		allowed = nil
	}
	noneSelected := func(path string) (version string) { return "none" }
	_, err := modload.QueryPackages(ctx, m.Path, m.Version, noneSelected, allowed)
	tooNew, ok := err.(*gover.TooNewError)
	if !ok {
		return module.Version{}, "", false
	}
	m.Path, m.Version, _ = strings.Cut(tooNew.What, "@")
	return m, tooNew.GoVersion, true
}
