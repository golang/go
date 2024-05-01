// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package toolchain implements dynamic switching of Go toolchains.
package toolchain

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"go/build"
	"io/fs"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/gover"
	"cmd/go/internal/modfetch"
	"cmd/go/internal/modload"
	"cmd/go/internal/run"
	"cmd/go/internal/work"

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

var counterErrorsInvalidToolchainInFile = base.NewCounter("go/errors:invalid-toolchain-in-file")

// Select invokes a different Go toolchain if directed by
// the GOTOOLCHAIN environment variable or the user's configuration
// or go.mod file.
// It must be called early in startup.
// See https://go.dev/doc/toolchain#select.
func Select() {
	log.SetPrefix("go: ")
	defer log.SetPrefix("")

	if !modload.WillBeEnabled() {
		return
	}

	// As a special case, let "go env GOTOOLCHAIN" and "go env -w GOTOOLCHAIN=..."
	// be handled by the local toolchain, since an older toolchain may not understand it.
	// This provides an easy way out of "go env -w GOTOOLCHAIN=go1.19" and makes
	// sure that "go env GOTOOLCHAIN" always prints the local go command's interpretation of it.
	// We look for these specific command lines in order to avoid mishandling
	//
	//	GOTOOLCHAIN=go1.999 go env -newflag GOTOOLCHAIN
	//
	// where -newflag is a flag known to Go 1.999 but not known to us.
	if (len(os.Args) == 3 && os.Args[1] == "env" && os.Args[2] == "GOTOOLCHAIN") ||
		(len(os.Args) == 4 && os.Args[1] == "env" && os.Args[2] == "-w" && strings.HasPrefix(os.Args[3], "GOTOOLCHAIN=")) {
		return
	}

	// As a special case, let "go env GOMOD" and "go env GOWORK" be handled by
	// the local toolchain. Users expect to be able to look up GOMOD and GOWORK
	// since the go.mod and go.work file need to be determined to determine
	// the minimum toolchain. See issue #61455.
	if len(os.Args) == 3 && os.Args[1] == "env" && (os.Args[2] == "GOMOD" || os.Args[2] == "GOWORK") {
		return
	}

	// Interpret GOTOOLCHAIN to select the Go toolchain to run.
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

	// Note: minToolchain is what https://go.dev/doc/toolchain#select calls the default toolchain.
	minToolchain := gover.LocalToolchain()
	minVers := gover.Local()
	var mode string
	if gotoolchain == "auto" {
		mode = "auto"
	} else if gotoolchain == "path" {
		mode = "path"
	} else {
		min, suffix, plus := strings.Cut(gotoolchain, "+") // go1.2.3+auto
		if min != "local" {
			v := gover.FromToolchain(min)
			if v == "" {
				if plus {
					base.Fatalf("invalid GOTOOLCHAIN %q: invalid minimum toolchain %q", gotoolchain, min)
				}
				base.Fatalf("invalid GOTOOLCHAIN %q", gotoolchain)
			}
			minToolchain = min
			minVers = v
		}
		if plus && suffix != "auto" && suffix != "path" {
			base.Fatalf("invalid GOTOOLCHAIN %q: only version suffixes are +auto and +path", gotoolchain)
		}
		mode = suffix
	}

	gotoolchain = minToolchain
	if (mode == "auto" || mode == "path") && !goInstallVersion() {
		// Read go.mod to find new minimum and suggested toolchain.
		file, goVers, toolchain := modGoToolchain()
		gover.Startup.AutoFile = file
		if toolchain == "default" {
			// "default" means always use the default toolchain,
			// which is already set, so nothing to do here.
			// Note that if we have Go 1.21 installed originally,
			// GOTOOLCHAIN=go1.30.0+auto or GOTOOLCHAIN=go1.30.0,
			// and the go.mod  says "toolchain default", we use Go 1.30, not Go 1.21.
			// That is, default overrides the "auto" part of the calculation
			// but not the minimum that the user has set.
			// Of course, if the go.mod also says "go 1.35", using Go 1.30
			// will provoke an error about the toolchain being too old.
			// That's what people who use toolchain default want:
			// only ever use the toolchain configured by the user
			// (including its environment and go env -w file).
			gover.Startup.AutoToolchain = toolchain
		} else {
			if toolchain != "" {
				// Accept toolchain only if it is > our min.
				// (If it is equal, then min satisfies it anyway: that can matter if min
				// has a suffix like "go1.21.1-foo" and toolchain is "go1.21.1".)
				toolVers := gover.FromToolchain(toolchain)
				if toolVers == "" || (!strings.HasPrefix(toolchain, "go") && !strings.Contains(toolchain, "-go")) {
					counterErrorsInvalidToolchainInFile.Inc()
					base.Fatalf("invalid toolchain %q in %s", toolchain, base.ShortPath(file))
				}
				if gover.Compare(toolVers, minVers) > 0 {
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

	counterSelectExec.Inc()
	Exec(gotoolchain)
}

var counterSelectExec = base.NewCounter("go/toolchain/select-exec")

// TestVersionSwitch is set in the test go binary to the value in $TESTGO_VERSION_SWITCH.
// Valid settings are:
//
//	"switch" - simulate version switches by reinvoking the test go binary with a different TESTGO_VERSION.
//	"mismatch" - like "switch" but forget to set TESTGO_VERSION, so it looks like we invoked a mismatched toolchain
//	"loop" - like "mismatch" but forget the target check, causing a toolchain switching loop
var TestVersionSwitch string

// Exec invokes the specified Go toolchain or else prints an error and exits the process.
// If $GOTOOLCHAIN is set to path or min+path, Exec only considers the PATH
// as a source of Go toolchains. Otherwise Exec tries the PATH but then downloads
// a toolchain if necessary.
func Exec(gotoolchain string) {
	log.SetPrefix("go: ")

	writeBits = sysWriteBits()

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
	if exe, err := cfg.LookPath(gotoolchain); err == nil {
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

	srcUGoMod := filepath.Join(dir, "src/_go.mod")
	srcGoMod := filepath.Join(dir, "src/go.mod")
	if size(srcGoMod) != size(srcUGoMod) {
		err := filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if path == srcUGoMod {
				// Leave for last, in case we are racing with another go command.
				return nil
			}
			if pdir, name := filepath.Split(path); name == "_go.mod" {
				if err := raceSafeCopy(path, pdir+"go.mod"); err != nil {
					return err
				}
			}
			return nil
		})
		// Handle src/go.mod; this is the signal to other racing go commands
		// that everything is okay and they can skip this step.
		if err == nil {
			err = raceSafeCopy(srcUGoMod, srcGoMod)
		}
		if err != nil {
			base.Fatalf("download %s: %v", gotoolchain, err)
		}
	}

	// Reinvoke the go command.
	execGoToolchain(gotoolchain, dir, filepath.Join(dir, "bin/go"))
}

func size(path string) int64 {
	info, err := os.Stat(path)
	if err != nil {
		return -1
	}
	return info.Size()
}

var writeBits fs.FileMode

// raceSafeCopy copies the file old to the file new, being careful to ensure
// that if multiple go commands call raceSafeCopy(old, new) at the same time,
// they don't interfere with each other: both will succeed and return and
// later observe the correct content in new. Like in the build cache, we arrange
// this by opening new without truncation and then writing the content.
// Both go commands can do this simultaneously and will write the same thing
// (old never changes content).
func raceSafeCopy(old, new string) error {
	oldInfo, err := os.Stat(old)
	if err != nil {
		return err
	}
	newInfo, err := os.Stat(new)
	if err == nil && newInfo.Size() == oldInfo.Size() {
		return nil
	}
	data, err := os.ReadFile(old)
	if err != nil {
		return err
	}
	// The module cache has unwritable directories by default.
	// Restore the user write bit in the directory so we can create
	// the new go.mod file. We clear it again at the end on a
	// best-effort basis (ignoring failures).
	dir := filepath.Dir(old)
	info, err := os.Stat(dir)
	if err != nil {
		return err
	}
	if err := os.Chmod(dir, info.Mode()|writeBits); err != nil {
		return err
	}
	defer os.Chmod(dir, info.Mode())
	// Note: create the file writable, so that a racing go command
	// doesn't get an error before we store the actual data.
	f, err := os.OpenFile(new, os.O_CREATE|os.O_WRONLY, writeBits&^0o111)
	if err != nil {
		// If OpenFile failed because a racing go command completed our work
		// (and then OpenFile failed because the directory or file is now read-only),
		// count that as a success.
		if size(old) == size(new) {
			return nil
		}
		return err
	}
	defer os.Chmod(new, oldInfo.Mode())
	if _, err := f.Write(data); err != nil {
		f.Close()
		return err
	}
	return f.Close()
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

// goInstallVersion reports whether the command line is go install m@v or go run m@v.
// If so, Select must not read the go.mod or go.work file in "auto" or "path" mode.
func goInstallVersion() bool {
	// Note: We assume there are no flags between 'go' and 'install' or 'run'.
	// During testing there are some debugging flags that are accepted
	// in that position, but in production go binaries there are not.
	if len(os.Args) < 3 {
		return false
	}

	var cmdFlags *flag.FlagSet
	switch os.Args[1] {
	default:
		// Command doesn't support a pkg@version as the main module.
		return false
	case "install":
		cmdFlags = &work.CmdInstall.Flag
	case "run":
		cmdFlags = &run.CmdRun.Flag
	}

	// The modcachrw flag is unique, in that it affects how we fetch the
	// requested module to even figure out what toolchain it needs.
	// We need to actually set it before we check the toolchain version.
	// (See https://go.dev/issue/64282.)
	modcacherwFlag := cmdFlags.Lookup("modcacherw")
	if modcacherwFlag == nil {
		base.Fatalf("internal error: modcacherw flag not registered for command")
	}
	modcacherwVal, ok := modcacherwFlag.Value.(interface {
		IsBoolFlag() bool
		flag.Value
	})
	if !ok || !modcacherwVal.IsBoolFlag() {
		base.Fatalf("internal error: modcacherw is not a boolean flag")
	}

	// Make a best effort to parse the command's args to find the pkg@version
	// argument and the -modcacherw flag.
	var (
		pkgArg         string
		modcacherwSeen bool
	)
	for args := os.Args[2:]; len(args) > 0; {
		a := args[0]
		args = args[1:]
		if a == "--" {
			if len(args) == 0 {
				return false
			}
			pkgArg = args[0]
			break
		}

		a, ok := strings.CutPrefix(a, "-")
		if !ok {
			// Not a flag argument. Must be a package.
			pkgArg = a
			break
		}
		a = strings.TrimPrefix(a, "-") // Treat --flag as -flag.

		name, val, hasEq := strings.Cut(a, "=")

		if name == "modcacherw" {
			if !hasEq {
				val = "true"
			}
			if err := modcacherwVal.Set(val); err != nil {
				return false
			}
			modcacherwSeen = true
			continue
		}

		if hasEq {
			// Already has a value; don't bother parsing it.
			continue
		}

		f := run.CmdRun.Flag.Lookup(a)
		if f == nil {
			// We don't know whether this flag is a boolean.
			if os.Args[1] == "run" {
				// We don't know where to find the pkg@version argument.
				// For run, the pkg@version can be anywhere on the command line,
				// because it is preceded by run flags and followed by arguments to the
				// program being run. Since we don't know whether this flag takes
				// an argument, we can't reliably identify the end of the run flags.
				// Just give up and let the user clarify using the "=" form..
				return false
			}

			// We would like to let 'go install -newflag pkg@version' work even
			// across a toolchain switch. To make that work, assume by default that
			// the pkg@version is the last argument and skip the remaining args unless
			// we spot a plausible "-modcacherw" flag.
			for len(args) > 0 {
				a := args[0]
				name, _, _ := strings.Cut(a, "=")
				if name == "-modcacherw" || name == "--modcacherw" {
					break
				}
				if len(args) == 1 && !strings.HasPrefix(a, "-") {
					pkgArg = a
				}
				args = args[1:]
			}
			continue
		}

		if bf, ok := f.Value.(interface{ IsBoolFlag() bool }); !ok || !bf.IsBoolFlag() {
			// The next arg is the value for this flag. Skip it.
			args = args[1:]
			continue
		}
	}

	if !strings.Contains(pkgArg, "@") || build.IsLocalImport(pkgArg) || filepath.IsAbs(pkgArg) {
		return false
	}
	path, version, _ := strings.Cut(pkgArg, "@")
	if path == "" || version == "" || gover.IsToolchain(path) {
		return false
	}

	if !modcacherwSeen && base.InGOFLAGS("-modcacherw") {
		fs := flag.NewFlagSet("goInstallVersion", flag.ExitOnError)
		fs.Var(modcacherwVal, "modcacherw", modcacherwFlag.Usage)
		base.SetFromGOFLAGS(fs)
	}

	// It would be correct to simply return true here, bypassing use
	// of the current go.mod or go.work, and let "go run" or "go install"
	// do the rest, including a toolchain switch.
	// Our goal instead is, since we have gone to the trouble of handling
	// unknown flags to some degree, to run the switch now, so that
	// these commands can switch to a newer toolchain directed by the
	// go.mod which may actually understand the flag.
	// This was brought up during the go.dev/issue/57001 proposal discussion
	// and may end up being common in self-contained "go install" or "go run"
	// command lines if we add new flags in the future.

	// Set up modules without an explicit go.mod, to download go.mod.
	modload.ForceUseModules = true
	modload.RootMode = modload.NoRoot
	modload.Init()
	defer modload.Reset()

	// See internal/load.PackagesAndErrorsOutsideModule
	ctx := context.Background()
	allowed := modload.CheckAllowed
	if modload.IsRevisionQuery(path, version) {
		// Don't check for retractions if a specific revision is requested.
		allowed = nil
	}
	noneSelected := func(path string) (version string) { return "none" }
	_, err := modload.QueryPackages(ctx, path, version, noneSelected, allowed)
	if errors.Is(err, gover.ErrTooNew) {
		// Run early switch, same one go install or go run would eventually do,
		// if it understood all the command-line flags.
		SwitchOrFatal(ctx, err)
	}

	return true // pkg@version found
}
