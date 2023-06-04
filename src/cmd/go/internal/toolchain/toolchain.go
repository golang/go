// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package toolchain implements dynamic switching of Go toolchains.
package toolchain

import (
	"context"
	"errors"
	"fmt"
	"go/build"
	"io/fs"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/gover"
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

	// targetEnv is a special environment variable set to the expected
	// toolchain version during the toolchain switch by the parent
	// process and cleared in the child process. When set, that indicates
	// to the child to confirm that it provides the expected toolchain version.
	targetEnv = "GOTOOLCHAIN_INTERNAL_SWITCH_VERSION"

	// countEnv is a special environment variable
	// that is incremented during each toolchain switch, to detect loops.
	// It is cleared before invoking programs in 'go run', 'go test', 'go generate', and 'go tool'
	// by invoking them in an environment filtered with FilterEnv,
	// so user programs should not see this in their environment.
	countEnv = "GOTOOLCHAIN_INTERNAL_SWITCH_COUNT"

	// maxSwitch is the maximum toolchain switching depth.
	// Most uses should never see more than three.
	// (Perhaps one for the initial GOTOOLCHAIN dispatch,
	// a second for go get doing an upgrade, and a third if
	// for some reason the chosen upgrade version is too small
	// by a little.)
	// When the count reaches maxSwitch - 10, we start logging
	// the switched versions for debugging before crashing with
	// a fatal error upon reaching maxSwitch.
	// That should be enough to see the repetition.
	maxSwitch = 100
)

// FilterEnv returns a copy of env with internal GOTOOLCHAIN environment
// variables filtered out.
func FilterEnv(env []string) []string {
	// Note: Don't need to filter out targetEnv because Switch does that.
	var out []string
	for _, e := range env {
		if strings.HasPrefix(e, countEnv+"=") {
			continue
		}
		out = append(out, e)
	}
	return out
}

// Switch invokes a different Go toolchain if directed by
// the GOTOOLCHAIN environment variable or the user's configuration
// or go.mod file.
// It must be called early in startup.
func Switch() {
	log.SetPrefix("go: ")
	defer log.SetPrefix("")

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

	minToolchain := gover.LocalToolchain()
	minVers := gover.Local()
	if min, mode, ok := strings.Cut(gotoolchain, "+"); ok { // go1.2.3+auto
		v := gover.FromToolchain(min)
		if v == "" {
			base.Fatalf("invalid GOTOOLCHAIN %q: invalid minimum toolchain %q", gotoolchain, min)
		}
		minToolchain = min
		minVers = v
		if mode != "auto" && mode != "path" {
			base.Fatalf("invalid GOTOOLCHAIN %q: only version suffixes are +auto and +path", gotoolchain)
		}
		gotoolchain = mode
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

	// If we are invoked as a target toolchain, confirm that
	// we provide the expected version and then run.
	// This check is delayed until after the handling of auto and path
	// so that we have initialized gover.Startup for use in error messages.
	if target := os.Getenv(targetEnv); target != "" && TestVersionSwitch != "loop" {
		if gover.LocalToolchain() != target {
			base.Fatalf("toolchain %v invoked to provide %v", gover.LocalToolchain(), target)
		}
		os.Unsetenv(targetEnv)

		// Note: It is tempting to check that if gotoolchain != "local"
		// then target == gotoolchain here, as a sanity check that
		// the child has made the same version determination as the parent.
		// This turns out not always to be the case. Specifically, if we are
		// running Go 1.21 with GOTOOLCHAIN=go1.22+auto, which invokes
		// Go 1.22, then 'go get go@1.23.0' or 'go get needs_go_1_23'
		// will invoke Go 1.23, but as the Go 1.23 child the reason for that
		// will not be apparent here: it will look like we should be using Go 1.22.
		// We rely on the targetEnv being set to know not to downgrade.
		// A longer term problem with the sanity check is that the exact details
		// may change over time: there may be other reasons that a future Go
		// version might invoke an older one, and the older one won't know why.
		// Best to just accept that we were invoked to provide a specific toolchain
		// (which we just checked) and leave it at that.
		return
	}

	if gotoolchain == "local" || gotoolchain == gover.LocalToolchain() {
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
	fetch := autoToolchains
	if !HasAuto() {
		fetch = pathToolchains
	}
	list, err := fetch(ctx)
	if err != nil {
		return "", err
	}
	return newerToolchain(version, list)
}

// autoToolchains returns the list of toolchain versions available to GOTOOLCHAIN=auto or =min+auto mode.
func autoToolchains(ctx context.Context) ([]string, error) {
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
		return nil, err
	}
	return versions.List, nil
}

// pathToolchains returns the list of toolchain versions available to GOTOOLCHAIN=path or =min+path mode.
func pathToolchains(ctx context.Context) ([]string, error) {
	have := make(map[string]bool)
	var list []string
	for _, dir := range pathDirs() {
		if dir == "" || !filepath.IsAbs(dir) {
			// Refuse to use local directories in $PATH (hard-coding exec.ErrDot).
			continue
		}
		entries, err := os.ReadDir(dir)
		if err != nil {
			continue
		}
		for _, de := range entries {
			if de.IsDir() || !strings.HasPrefix(de.Name(), "go1.") {
				continue
			}
			info, err := de.Info()
			if err != nil {
				continue
			}
			v, ok := pathVersion(dir, de, info)
			if !ok || !strings.HasPrefix(v, "1.") || have[v] {
				continue
			}
			have[v] = true
			list = append(list, v)
		}
	}
	sort.Slice(list, func(i, j int) bool {
		return gover.Compare(list[i], list[j]) < 0
	})
	return list, nil
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

// TestVersionSwitch is set in the test go binary to the value in $TESTGO_VERSION_SWITCH.
// Valid settings are:
//
//	"switch" - simulate version switches by reinvoking the test go binary with a different TESTGO_VERSION.
//	"mismatch" - like "switch" but forget to set TESTGO_VERSION, so it looks like we invoked a mismatched toolchain
//	"loop" - like "switch" but
var TestVersionSwitch string

// SwitchTo invokes the specified Go toolchain or else prints an error and exits the process.
// If $GOTOOLCHAIN is set to path or min+path, SwitchTo only considers the PATH
// as a source of Go toolchains. Otherwise SwitchTo tries the PATH but then downloads
// a toolchain if necessary.
func SwitchTo(gotoolchain string) {
	log.SetPrefix("go: ")

	count, _ := strconv.Atoi(os.Getenv(countEnv))
	if count >= maxSwitch-10 {
		fmt.Fprintf(os.Stderr, "go: switching from go%v to %v [depth %d]\n", gover.Local(), gotoolchain, count)
	}
	if count >= maxSwitch {
		base.Fatalf("too many toolchain switches")
	}
	os.Setenv(countEnv, fmt.Sprint(count+1))

	env := cfg.Getenv("GOTOOLCHAIN")
	pathOnly := env == "path" || strings.HasSuffix(env, "+path")

	// For testing, if TESTGO_VERSION is already in use
	// (only happens in the cmd/go test binary)
	// and TESTGO_VERSION_SWITCH=switch is set,
	// "switch" toolchains by changing TESTGO_VERSION
	// and reinvoking the current binary.
	// The special cases =loop and =mismatch skip the
	// setting of TESTGO_VERSION so that it looks like we
	// accidentally invoked the wrong toolchain,
	// to test detection of that failure mode.
	switch TestVersionSwitch {
	case "switch":
		os.Setenv("TESTGO_VERSION", gotoolchain)
		fallthrough
	case "loop", "mismatch":
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
	m := module.Version{
		Path:    gotoolchainModule,
		Version: gotoolchainVersion + "-" + gotoolchain + "." + runtime.GOOS + "-" + runtime.GOARCH,
	}
	dir, err := modfetch.Download(context.Background(), m)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			base.Fatalf("download %s for %s/%s: toolchain not available", gotoolchain, runtime.GOOS, runtime.GOARCH)
		}
		base.Fatalf("download %s: %v", gotoolchain, err)
	}

	// On first use after download, set the execute bits on the commands
	// so that we can run them. Note that multiple go commands might be
	// doing this at the same time, but if so no harm done.
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
func goInstallVersion() (m module.Version, goVers string, found bool) {
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
	if tooNew := (*gover.TooNewError)(nil); errors.As(err, &tooNew) {
		m.Path, m.Version, _ = strings.Cut(tooNew.What, "@")
		return m, tooNew.GoVersion, true
	}

	// QueryPackages succeeded, or it failed for a reason other than
	// this Go toolchain being too old for the modules encountered.
	// Either way, we identified the m@v on the command line,
	// so return found == true so the caller does not fall back to
	// consulting go.mod.
	return m, "", true
}
