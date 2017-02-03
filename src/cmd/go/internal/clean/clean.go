// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package clean implements the ``go clean'' command.
package clean

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/load"
	"cmd/go/internal/work"
)

var CmdClean = &base.Command{
	UsageLine: "clean [-i] [-r] [-n] [-x] [build flags] [packages]",
	Short:     "remove object files",
	Long: `
Clean removes object files from package source directories.
The go command builds most objects in a temporary directory,
so go clean is mainly concerned with object files left by other
tools or by manual invocations of go build.

Specifically, clean removes the following files from each of the
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

For more about build flags, see 'go help build'.

For more about specifying packages, see 'go help packages'.
	`,
}

var cleanI bool // clean -i flag
var cleanR bool // clean -r flag

func init() {
	// break init cycle
	CmdClean.Run = runClean

	CmdClean.Flag.BoolVar(&cleanI, "i", false, "")
	CmdClean.Flag.BoolVar(&cleanR, "r", false, "")
	// -n and -x are important enough to be
	// mentioned explicitly in the docs but they
	// are part of the build flags.

	work.AddBuildFlags(CmdClean)
}

func runClean(cmd *base.Command, args []string) {
	for _, pkg := range load.PackagesAndErrors(args) {
		clean(pkg)
	}
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
		base.Errorf("can't load package: %v", p.Error)
		return
	}
	dirs, err := ioutil.ReadDir(p.Dir)
	if err != nil {
		base.Errorf("go clean %s: %v", p.Dir, err)
		return
	}

	var b work.Builder
	b.Print = fmt.Print

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
		)
	}

	// Remove package test executables.
	allRemove = append(allRemove,
		elem+".test",
		elem+".test.exe",
	)

	// Remove a potential executable for each .go file in the directory that
	// is not part of the directory's package.
	for _, dir := range dirs {
		name := dir.Name()
		if packageFile[name] {
			continue
		}
		if !dir.IsDir() && strings.HasSuffix(name, ".go") {
			// TODO(adg,rsc): check that this .go file is actually
			// in "package main", and therefore capable of building
			// to an executable file.
			base := name[:len(name)-len(".go")]
			allRemove = append(allRemove, base, base+".exe")
		}
	}

	if cfg.BuildN || cfg.BuildX {
		b.Showcmd(p.Dir, "rm -f %s", strings.Join(allRemove, " "))
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
				if cfg.BuildN || cfg.BuildX {
					b.Showcmd(p.Dir, "rm -r %s", name)
					if cfg.BuildN {
						continue
					}
				}
				if err := os.RemoveAll(filepath.Join(p.Dir, name)); err != nil {
					base.Errorf("go clean: %v", err)
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

	if cleanI && p.Internal.Target != "" {
		if cfg.BuildN || cfg.BuildX {
			b.Showcmd("", "rm -f %s", p.Internal.Target)
		}
		if !cfg.BuildN {
			removeFile(p.Internal.Target)
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
	if base.ToolIsWindows {
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
	base.Errorf("go clean: %v", err)
}
