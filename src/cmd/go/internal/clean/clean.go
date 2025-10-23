// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package clean implements the “go clean” command.
package clean

import (
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"

	"cmd/go/internal/base"
	"cmd/go/internal/cache"
	"cmd/go/internal/cfg"
	"cmd/go/internal/load"
	"cmd/go/internal/lockedfile"
	"cmd/go/internal/modfetch"
	"cmd/go/internal/modload"
	"cmd/go/internal/str"
	"cmd/go/internal/work"
)

var CmdClean = &base.Command{
	UsageLine: "go clean [-i] [-r] [-cache] [-testcache] [-modcache] [-fuzzcache] [build flags] [packages]",
	Short:     "remove object files and cached files",
	Long: `
Clean removes object files from package source directories.
The go command builds most objects in a temporary directory,
so go clean is mainly concerned with object files left by other
tools or by manual invocations of go build.

If a package argument is given or the -i or -r flag is set,
clean removes the following files from each of the
source directories corresponding to the import paths:

	_obj/            old object directory, left from Makefiles
	_test/           old test directory, left from Makefiles
	_testmain.go     old gotest file, left from Makefiles
	test.out         old test log, left from Makefiles
	build.out        old test log, left from Makefiles
	*.[568ao]        object files, left from Makefiles

	DIR(.exe)        from go build
	DIR.test(.exe)   from go test -c
	MAINFILE(.exe)   from go build MAINFILE.go
	*.so             from SWIG

In the list, DIR represents the final path element of the
directory, and MAINFILE is the base name of any Go source
file in the directory that is not included when building
the package.

The -i flag causes clean to remove the corresponding installed
archive or binary (what 'go install' would create).

The -n flag causes clean to print the remove commands it would execute,
but not run them.

The -r flag causes clean to be applied recursively to all the
dependencies of the packages named by the import paths.

The -x flag causes clean to print remove commands as it executes them.

The -cache flag causes clean to remove the entire go build cache.

The -testcache flag causes clean to expire all test results in the
go build cache.

The -modcache flag causes clean to remove the entire module
download cache, including unpacked source code of versioned
dependencies.

The -fuzzcache flag causes clean to remove files stored in the Go build
cache for fuzz testing. The fuzzing engine caches files that expand
code coverage, so removing them may make fuzzing less effective until
new inputs are found that provide the same coverage. These files are
distinct from those stored in testdata directory; clean does not remove
those files.

For more about build flags, see 'go help build'.

For more about specifying packages, see 'go help packages'.
	`,
}

var (
	cleanI         bool // clean -i flag
	cleanR         bool // clean -r flag
	cleanCache     bool // clean -cache flag
	cleanFuzzcache bool // clean -fuzzcache flag
	cleanModcache  bool // clean -modcache flag
	cleanTestcache bool // clean -testcache flag
)

func init() {
	// break init cycle
	CmdClean.Run = runClean

	CmdClean.Flag.BoolVar(&cleanI, "i", false, "")
	CmdClean.Flag.BoolVar(&cleanR, "r", false, "")
	CmdClean.Flag.BoolVar(&cleanCache, "cache", false, "")
	CmdClean.Flag.BoolVar(&cleanFuzzcache, "fuzzcache", false, "")
	CmdClean.Flag.BoolVar(&cleanModcache, "modcache", false, "")
	CmdClean.Flag.BoolVar(&cleanTestcache, "testcache", false, "")

	// -n and -x are important enough to be
	// mentioned explicitly in the docs but they
	// are part of the build flags.

	work.AddBuildFlags(CmdClean, work.OmitBuildOnlyFlags)
}

func runClean(ctx context.Context, cmd *base.Command, args []string) {
	moduleLoaderState := modload.NewState()
	modload.InitWorkfile(moduleLoaderState)
	if len(args) > 0 {
		cacheFlag := ""
		switch {
		case cleanCache:
			cacheFlag = "-cache"
		case cleanTestcache:
			cacheFlag = "-testcache"
		case cleanFuzzcache:
			cacheFlag = "-fuzzcache"
		case cleanModcache:
			cacheFlag = "-modcache"
		}
		if cacheFlag != "" {
			base.Fatalf("go: clean %s cannot be used with package arguments", cacheFlag)
		}
	}

	// golang.org/issue/29925: only load packages before cleaning if
	// either the flags and arguments explicitly imply a package,
	// or no other target (such as a cache) was requested to be cleaned.
	cleanPkg := len(args) > 0 || cleanI || cleanR
	if (!modload.Enabled(moduleLoaderState) || modload.HasModRoot(moduleLoaderState)) &&
		!cleanCache && !cleanModcache && !cleanTestcache && !cleanFuzzcache {
		cleanPkg = true
	}

	if cleanPkg {
		for _, pkg := range load.PackagesAndErrors(moduleLoaderState, ctx, load.PackageOpts{}, args) {
			clean(pkg)
		}
	}

	sh := work.NewShell("", &load.TextPrinter{Writer: os.Stdout})

	if cleanCache {
		dir, _, err := cache.DefaultDir()
		if err != nil {
			base.Fatal(err)
		}
		if dir != "off" {
			// Remove the cache subdirectories but not the top cache directory.
			// The top cache directory may have been created with special permissions
			// and not something that we want to remove. Also, we'd like to preserve
			// the access log for future analysis, even if the cache is cleared.
			subdirs, _ := filepath.Glob(filepath.Join(str.QuoteGlob(dir), "[0-9a-f][0-9a-f]"))
			printedErrors := false
			if len(subdirs) > 0 {
				if err := sh.RemoveAll(subdirs...); err != nil && !printedErrors {
					printedErrors = true
					base.Error(err)
				}
			}

			logFile := filepath.Join(dir, "log.txt")
			if err := sh.RemoveAll(logFile); err != nil && !printedErrors {
				printedErrors = true
				base.Error(err)
			}
		}
	}

	if cleanTestcache && !cleanCache {
		// Instead of walking through the entire cache looking for test results,
		// we write a file to the cache indicating that all test results from before
		// right now are to be ignored.
		dir, _, err := cache.DefaultDir()
		if err != nil {
			base.Fatal(err)
		}
		if dir != "off" {
			f, err := lockedfile.Edit(filepath.Join(dir, "testexpire.txt"))
			if err == nil {
				now := time.Now().UnixNano()
				buf, _ := io.ReadAll(f)
				prev, _ := strconv.ParseInt(strings.TrimSpace(string(buf)), 10, 64)
				if now > prev {
					if err = f.Truncate(0); err == nil {
						if _, err = f.Seek(0, 0); err == nil {
							_, err = fmt.Fprintf(f, "%d\n", now)
						}
					}
				}
				if closeErr := f.Close(); err == nil {
					err = closeErr
				}
			}
			if err != nil {
				if _, statErr := os.Stat(dir); !os.IsNotExist(statErr) {
					base.Error(err)
				}
			}
		}
	}

	if cleanModcache {
		if cfg.GOMODCACHE == "" {
			base.Fatalf("go: cannot clean -modcache without a module cache")
		}
		if cfg.BuildN || cfg.BuildX {
			sh.ShowCmd("", "rm -rf %s", cfg.GOMODCACHE)
		}
		if !cfg.BuildN {
			if err := modfetch.RemoveAll(cfg.GOMODCACHE); err != nil {
				base.Error(err)

				// Add extra logging for the purposes of debugging #68087.
				// We're getting ENOTEMPTY errors on openbsd from RemoveAll.
				// Check for os.ErrExist, which can match syscall.ENOTEMPTY
				// and syscall.EEXIST, because syscall.ENOTEMPTY is not defined
				// on all platforms.
				if runtime.GOOS == "openbsd" && errors.Is(err, fs.ErrExist) {
					logFilesInGOMODCACHE()
				}
			}
		}
	}

	if cleanFuzzcache {
		fuzzDir := cache.Default().FuzzDir()
		if err := sh.RemoveAll(fuzzDir); err != nil {
			base.Error(err)
		}
	}
}

// logFilesInGOMODCACHE reports the file names and modes for the files in GOMODCACHE using base.Error.
func logFilesInGOMODCACHE() {
	var found []string
	werr := filepath.WalkDir(cfg.GOMODCACHE, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		var mode string
		info, err := d.Info()
		if err == nil {
			mode = info.Mode().String()
		} else {
			mode = fmt.Sprintf("<err: %s>", info.Mode())
		}
		found = append(found, fmt.Sprintf("%s (mode: %s)", path, mode))
		return nil
	})
	if werr != nil {
		base.Errorf("walking files in GOMODCACHE (for debugging go.dev/issue/68087): %v", werr)
	}
	base.Errorf("files in GOMODCACHE (for debugging go.dev/issue/68087):\n%s", strings.Join(found, "\n"))
}

var cleaned = map[*load.Package]bool{}

// TODO: These are dregs left by Makefile-based builds.
// Eventually, can stop deleting these.
var cleanDir = map[string]bool{
	"_test": true,
	"_obj":  true,
}

var cleanFile = map[string]bool{
	"_testmain.go": true,
	"test.out":     true,
	"build.out":    true,
	"a.out":        true,
}

var cleanExt = map[string]bool{
	".5":  true,
	".6":  true,
	".8":  true,
	".a":  true,
	".o":  true,
	".so": true,
}

func clean(p *load.Package) {
	if cleaned[p] {
		return
	}
	cleaned[p] = true

	if p.Dir == "" {
		base.Errorf("%v", p.Error)
		return
	}
	dirs, err := os.ReadDir(p.Dir)
	if err != nil {
		base.Errorf("go: %s: %v", p.Dir, err)
		return
	}

	sh := work.NewShell("", &load.TextPrinter{Writer: os.Stdout})

	packageFile := map[string]bool{}
	if p.Name != "main" {
		// Record which files are not in package main.
		// The others are.
		keep := func(list []string) {
			for _, f := range list {
				packageFile[f] = true
			}
		}
		keep(p.GoFiles)
		keep(p.CgoFiles)
		keep(p.TestGoFiles)
		keep(p.XTestGoFiles)
	}

	_, elem := filepath.Split(p.Dir)
	var allRemove []string

	// Remove dir-named executable only if this is package main.
	if p.Name == "main" {
		allRemove = append(allRemove,
			elem,
			elem+".exe",
			p.DefaultExecName(),
			p.DefaultExecName()+".exe",
		)
	}

	// Remove package test executables.
	allRemove = append(allRemove,
		elem+".test",
		elem+".test.exe",
		p.DefaultExecName()+".test",
		p.DefaultExecName()+".test.exe",
	)

	// Remove a potential executable, test executable for each .go file in the directory that
	// is not part of the directory's package.
	for _, dir := range dirs {
		name := dir.Name()
		if packageFile[name] {
			continue
		}

		if dir.IsDir() {
			continue
		}

		if base, found := strings.CutSuffix(name, "_test.go"); found {
			allRemove = append(allRemove, base+".test", base+".test.exe")
		}

		if base, found := strings.CutSuffix(name, ".go"); found {
			// TODO(adg,rsc): check that this .go file is actually
			// in "package main", and therefore capable of building
			// to an executable file.
			allRemove = append(allRemove, base, base+".exe")
		}
	}

	if cfg.BuildN || cfg.BuildX {
		sh.ShowCmd(p.Dir, "rm -f %s", strings.Join(allRemove, " "))
	}

	toRemove := map[string]bool{}
	for _, name := range allRemove {
		toRemove[name] = true
	}
	for _, dir := range dirs {
		name := dir.Name()
		if dir.IsDir() {
			// TODO: Remove once Makefiles are forgotten.
			if cleanDir[name] {
				if err := sh.RemoveAll(filepath.Join(p.Dir, name)); err != nil {
					base.Error(err)
				}
			}
			continue
		}

		if cfg.BuildN {
			continue
		}

		if cleanFile[name] || cleanExt[filepath.Ext(name)] || toRemove[name] {
			removeFile(filepath.Join(p.Dir, name))
		}
	}

	if cleanI && p.Target != "" {
		if cfg.BuildN || cfg.BuildX {
			sh.ShowCmd("", "rm -f %s", p.Target)
		}
		if !cfg.BuildN {
			removeFile(p.Target)
		}
	}

	if cleanR {
		for _, p1 := range p.Internal.Imports {
			clean(p1)
		}
	}
}

// removeFile tries to remove file f, if error other than file doesn't exist
// occurs, it will report the error.
func removeFile(f string) {
	err := os.Remove(f)
	if err == nil || os.IsNotExist(err) {
		return
	}
	// Windows does not allow deletion of a binary file while it is executing.
	if runtime.GOOS == "windows" {
		// Remove lingering ~ file from last attempt.
		if _, err2 := os.Stat(f + "~"); err2 == nil {
			os.Remove(f + "~")
		}
		// Try to move it out of the way. If the move fails,
		// which is likely, we'll try again the
		// next time we do an install of this binary.
		if err2 := os.Rename(f, f+"~"); err2 == nil {
			os.Remove(f + "~")
			return
		}
	}
	base.Error(err)
}
