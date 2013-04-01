// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
)

var cmdClean = &Command{
	UsageLine: "clean [-i] [-r] [-n] [-x] [packages]",
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

For more about specifying packages, see 'go help packages'.
	`,
}

var cleanI bool // clean -i flag
var cleanN bool // clean -n flag
var cleanR bool // clean -r flag
var cleanX bool // clean -x flag

func init() {
	// break init cycle
	cmdClean.Run = runClean

	cmdClean.Flag.BoolVar(&cleanI, "i", false, "")
	cmdClean.Flag.BoolVar(&cleanN, "n", false, "")
	cmdClean.Flag.BoolVar(&cleanR, "r", false, "")
	cmdClean.Flag.BoolVar(&cleanX, "x", false, "")
}

func runClean(cmd *Command, args []string) {
	for _, pkg := range packagesAndErrors(args) {
		clean(pkg)
	}
}

var cleaned = map[*Package]bool{}

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

func clean(p *Package) {
	if cleaned[p] {
		return
	}
	cleaned[p] = true

	if p.Dir == "" {
		errorf("can't load package: %v", p.Error)
		return
	}
	dirs, err := ioutil.ReadDir(p.Dir)
	if err != nil {
		errorf("go clean %s: %v", p.Dir, err)
		return
	}

	var b builder
	b.print = fmt.Print

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
	allRemove := []string{
		elem,
		elem + ".exe",
		elem + ".test",
		elem + ".test.exe",
	}
	for _, dir := range dirs {
		name := dir.Name()
		if packageFile[name] {
			continue
		}
		if !dir.IsDir() && strings.HasSuffix(name, ".go") {
			base := name[:len(name)-len(".go")]
			allRemove = append(allRemove, base, base+".exe")
		}
	}
	if cleanN || cleanX {
		b.showcmd(p.Dir, "rm -f %s", strings.Join(allRemove, " "))
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
				if cleanN || cleanX {
					b.showcmd(p.Dir, "rm -r %s", name)
					if cleanN {
						continue
					}
				}
				if err := os.RemoveAll(filepath.Join(p.Dir, name)); err != nil {
					errorf("go clean: %v", err)
				}
			}
			continue
		}

		if cleanN {
			continue
		}

		if cleanFile[name] || cleanExt[filepath.Ext(name)] || toRemove[name] {
			removeFile(filepath.Join(p.Dir, name))
		}
	}

	if cleanI && p.target != "" {
		if cleanN || cleanX {
			b.showcmd("", "rm -f %s", p.target)
		}
		if !cleanN {
			removeFile(p.target)
		}
	}

	if cleanI && p.usesSwig() {
		for _, f := range stringList(p.SwigFiles, p.SwigCXXFiles) {
			dir := p.swigDir(&buildContext)
			soname := p.swigSoname(f)
			target := filepath.Join(dir, soname)
			if cleanN || cleanX {
				b.showcmd("", "rm -f %s", target)
			}
			if !cleanN {
				removeFile(target)
			}
		}
	}

	if cleanR {
		for _, p1 := range p.imports {
			clean(p1)
		}
	}
}

// removeFile tries to remove file f, if error other than file doesn't exist
// occurs, it will report the error.
func removeFile(f string) {
	if err := os.Remove(f); err != nil && !os.IsNotExist(err) {
		errorf("go clean: %v", err)
	}
}
