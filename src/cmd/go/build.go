// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"errors"
	"fmt"
	"go/build"
	"io"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
)

// Break init cycles
func init() {
	cmdBuild.Run = runBuild
	cmdInstall.Run = runInstall
}

var cmdBuild = &Command{
	UsageLine: "build [-a] [-n] [-x] [importpath... | gofiles...]",
	Short:     "compile packages and dependencies",
	Long: `
Build compiles the packages named by the import paths,
along with their dependencies, but it does not install the results.

If the arguments are a list of .go files, build compiles them into
a package object or command executable named for the first
source file.

The -a flag forces rebuilding of packages that are already up-to-date.
The -n flag prints the commands but does not run them.
The -x flag prints the commands.

For more about import paths, see 'go help importpath'.

See also: go install, go get, go clean.
	`,
}

var buildA = cmdBuild.Flag.Bool("a", false, "")
var buildN = cmdBuild.Flag.Bool("n", false, "")
var buildX = cmdBuild.Flag.Bool("x", false, "")

func runBuild(cmd *Command, args []string) {
	var b builder
	b.init(*buildA, *buildN, *buildX)

	if len(args) > 0 && strings.HasSuffix(args[0], ".go") {
		b.do(b.action(modeInstall, modeBuild, goFilesPackage(args, "")))
		return
	}

	a := &action{}
	for _, p := range packages(args) {
		a.deps = append(a.deps, b.action(modeBuild, modeBuild, p))
	}
	b.do(a)
}

var cmdInstall = &Command{
	UsageLine: "install [-a] [-n] [-x] [importpath...]",
	Short:     "compile and install packages and dependencies",
	Long: `
Install compiles and installs the packages named by the import paths,
along with their dependencies.

The -a flag forces reinstallation of packages that are already up-to-date.
The -n flag prints the commands but does not run them.
The -x flag prints the commands.

For more about import paths, see 'go help importpath'.

See also: go build, go get, go clean.
	`,
}

var installA = cmdInstall.Flag.Bool("a", false, "")
var installN = cmdInstall.Flag.Bool("n", false, "")
var installX = cmdInstall.Flag.Bool("x", false, "")

func runInstall(cmd *Command, args []string) {
	var b builder
	b.init(*installA, *installN, *installX)
	a := &action{}
	for _, p := range packages(args) {
		a.deps = append(a.deps, b.action(modeInstall, modeInstall, p))
	}
	b.do(a)
}

// A builder holds global state about a build.
// It does not hold per-package state, because eventually we will
// build packages in parallel, and the builder will be shared.
type builder struct {
	work        string               // the temporary work directory (ends in filepath.Separator)
	aflag       bool                 // the -a flag
	nflag       bool                 // the -n flag
	xflag       bool                 // the -x flag
	arch        string               // e.g., "6"
	goroot      string               // the $GOROOT
	goarch      string               // the $GOARCH
	goos        string               // the $GOOS
	gobin       string               // the $GOBIN
	actionCache map[cacheKey]*action // a cache of already-constructed actions

	output    sync.Mutex
	scriptDir string // current directory in printed script
}

// An action represents a single action in the action graph.
type action struct {
	f func(*builder, *action) error // the action itself (nil = no-op)

	p          *Package  // the package this action works on
	deps       []*action // actions that must happen before this one
	done       bool      // whether the action is done (might have failed)
	failed     bool      // whether the action failed
	pkgdir     string    // the -I or -L argument to use when importing this package
	ignoreFail bool      // whether to run f even if dependencies fail

	// Results left for communication with other code.
	pkgobj string // the built .a file
	pkgbin string // the built a.out file, if one exists
}

// cacheKey is the key for the action cache.
type cacheKey struct {
	mode buildMode
	p    *Package
}

// buildMode specifies the build mode:
// are we just building things or also installing the results?
type buildMode int

const (
	modeBuild buildMode = iota
	modeInstall
)

func (b *builder) init(aflag, nflag, xflag bool) {
	var err error
	b.aflag = aflag
	b.nflag = nflag
	b.xflag = xflag
	b.actionCache = make(map[cacheKey]*action)
	b.goarch = build.DefaultContext.GOARCH
	b.goos = build.DefaultContext.GOOS
	b.goroot = build.Path[0].Path
	b.gobin = build.Path[0].BinDir()

	b.arch, err = build.ArchChar(b.goarch)
	if err != nil {
		fatalf("%s", err)
	}

	if nflag {
		b.work = "$WORK"
	} else {
		b.work, err = ioutil.TempDir("", "go-build")
		if err != nil {
			fatalf("%s", err)
		}
		if b.xflag {
			fmt.Printf("WORK=%s\n", b.work)
		}
		atexit(func() { os.RemoveAll(b.work) })
	}
}

// goFilesPackage creates a package for building a collection of Go files
// (typically named on the command line).  If target is given, the package
// target is target.  Otherwise, the target is named p.a for
// package p or named after the first Go file for package main.
func goFilesPackage(gofiles []string, target string) *Package {
	// Synthesize fake "directory" that only shows those two files,
	// to make it look like this is a standard package or
	// command directory.
	var dir []os.FileInfo
	for _, file := range gofiles {
		fi, err := os.Stat(file)
		if err != nil {
			fatalf("%s", err)
		}
		if fi.IsDir() {
			fatalf("%s is a directory, should be a Go file", file)
		}
		dir = append(dir, fi)
	}
	ctxt := build.DefaultContext
	ctxt.ReadDir = func(string) ([]os.FileInfo, error) { return dir, nil }
	pwd, _ := os.Getwd()
	pkg, err := scanPackage(&ctxt, &build.Tree{Path: "."}, "<command line>", "<command line>", pwd)
	if err != nil {
		fatalf("%s", err)
	}
	if target != "" {
		pkg.targ = target
	} else if pkg.Name == "main" {
		pkg.targ = gofiles[0][:len(gofiles[0])-len(".go")]
	} else {
		pkg.targ = pkg.Name + ".a"
	}
	pkg.ImportPath = "_/" + pkg.targ
	return pkg
}

// action returns the action for applying the given operation (mode) to the package.
// depMode is the action to use when building dependencies.
func (b *builder) action(mode buildMode, depMode buildMode, p *Package) *action {
	key := cacheKey{mode, p}
	a := b.actionCache[key]
	if a != nil {
		return a
	}

	a = &action{p: p, pkgdir: p.t.PkgDir()}
	if p.pkgdir != "" { // overrides p.t
		a.pkgdir = p.pkgdir
	}

	b.actionCache[key] = a

	switch mode {
	case modeBuild, modeInstall:
		for _, p1 := range p.imports {
			a.deps = append(a.deps, b.action(depMode, depMode, p1))
		}

		if !needInstall(p) && !b.aflag && allNop(a.deps) {
			return a
		}
		if p.Standard {
			switch p.ImportPath {
			case "builtin", "unsafe":
				// Fake packages - nothing to build.
				return a
			}
		}

		if mode == modeInstall {
			a.f = (*builder).install
			a.deps = []*action{b.action(modeBuild, depMode, p)}
			return a
		}

		a.f = (*builder).build
	}

	return a
}

func allNop(actions []*action) bool {
	for _, a := range actions {
		if a.f != nil {
			return false
		}
	}
	return true
}

// needInstall reports whether p needs to be built and installed.
// That is only true if some source file is newer than the installed package binary.
func needInstall(p *Package) bool {
	if p.targ == "" {
		return true
	}
	fi, err := os.Stat(p.targ)
	if err != nil {
		return true
	}
	t := fi.ModTime()

	srcss := [][]string{
		p.GoFiles,
		p.CFiles,
		p.SFiles,
		p.CgoFiles,
	}
	for _, srcs := range srcss {
		for _, src := range srcs {
			fi, err := os.Stat(filepath.Join(p.Dir, src))
			if err != nil {
				return true
			}
			if fi.ModTime().After(t) {
				return true
			}
		}
	}

	return false
}

// do runs the action graph rooted at a.
func (b *builder) do(a *action) {
	if a.done {
		return
	}
	for _, a1 := range a.deps {
		b.do(a1)
		if a1.failed {
			a.failed = true
			if !a.ignoreFail {
				a.done = true
				return
			}
		}
	}
	if a.f != nil {
		if err := a.f(b, a); err != nil {
			if err == errPrintedOutput {
				exitStatus = 2
			} else {
				errorf("%s", err)
			}
			a.failed = true
		}
	}
	a.done = true
}

// build is the action for building a single package or command.
func (b *builder) build(a *action) error {
	if b.nflag {
		// In -n mode, print a banner between packages.
		// The banner is five lines so that when changes to
		// different sections of the bootstrap script have to
		// be merged, the banners give patch something
		// to use to find its context.
		fmt.Printf("\n#\n# %s\n#\n\n", a.p.ImportPath)
	}
	obj := filepath.Join(b.work, filepath.FromSlash(a.p.ImportPath+"/_obj")) + string(filepath.Separator)
	if a.pkgobj == "" {
		a.pkgobj = filepath.Join(b.work, filepath.FromSlash(a.p.ImportPath+".a"))
	}

	// make build directory
	if err := b.mkdir(obj); err != nil {
		return err
	}

	var gofiles, cfiles, sfiles, objects []string
	gofiles = append(gofiles, a.p.GoFiles...)
	cfiles = append(cfiles, a.p.CFiles...)
	sfiles = append(sfiles, a.p.SFiles...)

	// run cgo
	if len(a.p.CgoFiles) > 0 {
		// In a package using cgo, cgo compiles the C and assembly files with gcc.  
		// There is one exception: runtime/cgo's job is to bridge the
		// cgo and non-cgo worlds, so it necessarily has files in both.
		// In that case gcc only gets the gcc_* files.
		var gccfiles []string
		if a.p.Standard && a.p.ImportPath == "runtime/cgo" {
			filter := func(files, nongcc, gcc []string) ([]string, []string) {
				for _, f := range files {
					if strings.HasPrefix(f, "gcc_") {
						gcc = append(gcc, f)
					} else {
						nongcc = append(nongcc, f)
					}
				}
				return nongcc, gcc
			}
			cfiles, gccfiles = filter(cfiles, cfiles[:0], gccfiles)
			sfiles, gccfiles = filter(sfiles, sfiles[:0], gccfiles)
		} else {
			gccfiles = append(cfiles, sfiles...)
			cfiles = nil
			sfiles = nil
		}

		outGo, outObj, err := b.cgo(a.p, obj, gccfiles)
		if err != nil {
			return err
		}
		objects = append(objects, outObj...)
		gofiles = append(gofiles, outGo...)
	}

	// prepare Go import path list
	inc := []string{}
	incMap := map[string]bool{}

	// work directory first
	inc = append(inc, "-I", b.work)
	incMap[b.work] = true
	incMap[build.Path[0].PkgDir()] = true // goroot
	incMap[""] = true                     // ignore empty strings

	// then build package directories of dependencies
	for _, a1 := range a.deps {
		if pkgdir := a1.pkgdir; !incMap[pkgdir] {
			incMap[pkgdir] = true
			inc = append(inc, "-I", pkgdir)
		}
	}

	// then installed package directories of dependencies
	for _, a1 := range a.deps {
		if pkgdir := a1.p.t.PkgDir(); !incMap[pkgdir] {
			incMap[pkgdir] = true
			inc = append(inc, "-I", pkgdir)
		}
	}

	// compile Go
	if len(gofiles) > 0 {
		out := "_go_.6"
		gcargs := []string{"-p", a.p.ImportPath}
		if a.p.Standard && a.p.ImportPath == "runtime" {
			// runtime compiles with a special 6g flag to emit
			// additional reflect type data.
			gcargs = append(gcargs, "-+")
		}
		if err := b.gc(a.p, obj+out, gcargs, inc, gofiles); err != nil {
			return err
		}
		objects = append(objects, out)
	}

	// copy .h files named for goos or goarch or goos_goarch
	// to names using GOOS and GOARCH.
	// For example, defs_linux_amd64.h becomes defs_GOOS_GOARCH.h.
	_goos_goarch := "_" + b.goos + "_" + b.goarch + ".h"
	_goos := "_" + b.goos + ".h"
	_goarch := "_" + b.goarch + ".h"
	for _, file := range a.p.HFiles {
		switch {
		case strings.HasSuffix(file, _goos_goarch):
			targ := file[:len(file)-len(_goos_goarch)] + "_GOOS_GOARCH.h"
			if err := b.copyFile(obj+targ, filepath.Join(a.p.Dir, file), 0666); err != nil {
				return err
			}
		case strings.HasSuffix(file, _goarch):
			targ := file[:len(file)-len(_goarch)] + "_GOARCH.h"
			if err := b.copyFile(obj+targ, filepath.Join(a.p.Dir, file), 0666); err != nil {
				return err
			}
		case strings.HasSuffix(file, _goos):
			targ := file[:len(file)-len(_goos)] + "_GOOS.h"
			if err := b.copyFile(obj+targ, filepath.Join(a.p.Dir, file), 0666); err != nil {
				return err
			}
		}
	}

	for _, file := range cfiles {
		out := file[:len(file)-len(".c")] + "." + b.arch
		if err := b.cc(a.p, obj, obj+out, file); err != nil {
			return err
		}
		objects = append(objects, out)
	}

	// assemble .s files
	for _, file := range sfiles {
		out := file[:len(file)-len(".s")] + "." + b.arch
		if err := b.asm(a.p, obj, obj+out, file); err != nil {
			return err
		}
		objects = append(objects, out)
	}

	// pack into archive
	if err := b.gopack(a.p, obj, a.pkgobj, objects); err != nil {
		return err
	}

	if a.p.Name == "main" {
		// command.
		// import paths for compiler are introduced by -I.
		// for linker, they are introduced by -L.
		for i := 0; i < len(inc); i += 2 {
			inc[i] = "-L"
		}
		a.pkgbin = obj + "a.out"
		if err := b.ld(a.p, a.pkgbin, inc, a.pkgobj); err != nil {
			return err
		}
	}

	return nil
}

// install is the action for installing a single package.
func (b *builder) install(a *action) error {
	a1 := a.deps[0]
	var src string
	var perm uint32
	if a1.pkgbin != "" {
		src = a1.pkgbin
		perm = 0777
	} else {
		src = a1.pkgobj
		perm = 0666
	}

	// make target directory
	dst := a.p.targ
	dir, _ := filepath.Split(dst)
	if dir != "" {
		if err := b.mkdir(dir); err != nil {
			return err
		}
	}

	return b.copyFile(dst, src, perm)
}

// copyFile is like 'cp src dst'.
func (b *builder) copyFile(dst, src string, perm uint32) error {
	if b.nflag || b.xflag {
		b.showcmd("", "cp %s %s", src, dst)
		if b.nflag {
			return nil
		}
	}

	sf, err := os.Open(src)
	if err != nil {
		return err
	}
	defer sf.Close()
	os.Remove(dst)
	df, err := os.OpenFile(dst, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, perm)
	if err != nil {
		return err
	}
	_, err = io.Copy(df, sf)
	df.Close()
	if err != nil {
		os.Remove(dst)
		return err
	}
	return nil
}

// fmtcmd formats a command in the manner of fmt.Sprintf but also:
//
//	If dir is non-empty and the script is not in dir right now,
//	fmtcmd inserts "cd dir\n" before the command.
//
//	fmtcmd replaces the value of b.work with $WORK.
//	fmtcmd replaces the value of b.goroot with $GOROOT.
//	fmtcmd replaces the value of b.gobin with $GOBIN.
//
//	fmtcmd replaces the name of the current directory with dot (.)
//	but only when it is at the beginning of a space-separated token.
//
func (b *builder) fmtcmd(dir string, format string, args ...interface{}) string {
	cmd := fmt.Sprintf(format, args...)
	if dir != "" {
		cmd = strings.Replace(" "+cmd, " "+dir, " .", -1)[1:]
		if b.scriptDir != dir {
			b.scriptDir = dir
			cmd = " cd " + dir + "\n" + cmd
		}
	}
	cmd = strings.Replace(cmd, b.work, "$WORK", -1)
	cmd = strings.Replace(cmd, b.gobin, "$GOBIN", -1)
	cmd = strings.Replace(cmd, b.goroot, "$GOROOT", -1)
	return cmd
}

// showcmd prints the given command to standard output
// for the implementation of -n or -x.
func (b *builder) showcmd(dir string, format string, args ...interface{}) {
	b.output.Lock()
	defer b.output.Unlock()
	fmt.Println(b.fmtcmd(dir, format, args...))
}

// showOutput prints "# desc" followed by the given output.
// The output is expected to contain references to 'dir', usually
// the source directory for the package that has failed to build.
// showOutput rewrites mentions of dir with a relative path to dir.
// This is usually shorter and more pleasant than the absolute path.
// For example, if fmt doesn't compile and we are in src/pkg/html,
// the output is
//
//	$ go build
//	# fmt
//	../fmt/print.go:1090: undefined: asdf
//	$
//
// instead of
//
//	$ go build
//	# fmt
//	/usr/gopher/go/src/pkg/fmt/print.go:1090: undefined: asdf
//	$
//
// showOutput also replaces references to the work directory with $WORK.
//
func (b *builder) showOutput(dir, desc, out string) {
	prefix := "# " + desc
	suffix := "\n" + out
	pwd, _ := os.Getwd()
	if reldir, err := filepath.Rel(pwd, dir); err == nil {
		suffix = strings.Replace(suffix, " "+dir, " "+reldir, -1)
		suffix = strings.Replace(suffix, "\n"+dir, "\n"+reldir, -1)
	}
	suffix = strings.Replace(suffix, " "+b.work, " $WORK", -1)

	b.output.Lock()
	defer b.output.Unlock()
	fmt.Print(prefix, suffix)
}

// errPrintedOutput is a special error indicating that a command failed
// but that it generated output as well, and that output has already
// been printed, so there's no point showing 'exit status 1' or whatever
// the wait status was.  The main executor, builder.do, knows not to
// print this error.
var errPrintedOutput = errors.New("already printed output - no need to show error")

// run runs the command given by cmdline in the directory dir.
// If the commnd fails, run prints information about the failure
// and returns a non-nil error.
func (b *builder) run(dir string, desc string, cmdline ...string) error {
	if b.nflag || b.xflag {
		b.showcmd(dir, "%s", strings.Join(cmdline, " "))
		if b.nflag {
			return nil
		}
	}

	var buf bytes.Buffer
	cmd := exec.Command(cmdline[0], cmdline[1:]...)
	cmd.Stdout = &buf
	cmd.Stderr = &buf
	cmd.Dir = dir
	// TODO: cmd.Env
	err := cmd.Run()
	if buf.Len() > 0 {
		out := buf.Bytes()
		if out[len(out)-1] != '\n' {
			out = append(out, '\n')
		}
		if desc == "" {
			desc = b.fmtcmd(dir, "%s", strings.Join(cmdline, " "))
		}
		b.showOutput(dir, desc, string(out))
		if err != nil {
			err = errPrintedOutput
		}
	}
	return err
}

// mkdir makes the named directory.
func (b *builder) mkdir(dir string) error {
	if b.nflag || b.xflag {
		b.showcmd("", "mkdir -p %s", dir)
		if b.nflag {
			return nil
		}
	}

	if err := os.MkdirAll(dir, 0777); err != nil {
		return err
	}
	return nil
}

// mkAbs returns an absolute path corresponding to
// evaluating f in the directory dir.
// We always pass absolute paths of source files so that
// the error messages will include the full path to a file
// in need of attention.
func mkAbs(dir, f string) string {
	// Leave absolute paths alone.
	// Also, during -n mode we use the pseudo-directory $WORK
	// instead of creating an actual work directory that won't be used.
	// Leave paths beginning with $WORK alone too.
	if filepath.IsAbs(f) || strings.HasPrefix(f, "$WORK") {
		return f
	}
	return filepath.Join(dir, f)
}

// gc runs the Go compiler in a specific directory on a set of files
// to generate the named output file. 
func (b *builder) gc(p *Package, ofile string, gcargs, importArgs []string, gofiles []string) error {
	args := []string{b.arch + "g", "-o", ofile}
	args = append(args, gcargs...)
	args = append(args, importArgs...)
	for _, f := range gofiles {
		args = append(args, mkAbs(p.Dir, f))
	}
	return b.run(p.Dir, p.ImportPath, args...)
}

// asm runs the assembler in a specific directory on a specific file
// to generate the named output file. 
func (b *builder) asm(p *Package, obj, ofile, sfile string) error {
	sfile = mkAbs(p.Dir, sfile)
	return b.run(p.Dir, p.ImportPath, b.arch+"a", "-I", obj, "-o", ofile, "-DGOOS_"+b.goos, "-DGOARCH_"+b.goarch, sfile)
}

// gopack runs the assembler in a specific directory to create
// an archive from a set of object files.
// typically it is run in the object directory.
func (b *builder) gopack(p *Package, objDir, afile string, ofiles []string) error {
	cmd := []string{"gopack", "grc"}
	cmd = append(cmd, mkAbs(objDir, afile))
	for _, f := range ofiles {
		cmd = append(cmd, mkAbs(objDir, f))
	}
	return b.run(p.Dir, p.ImportPath, cmd...)
}

// ld runs the linker to create a package starting at mainpkg.
func (b *builder) ld(p *Package, out string, importArgs []string, mainpkg string) error {
	return b.run(p.Dir, p.ImportPath, append(append([]string{b.arch + "l", "-o", out}, importArgs...), mainpkg)...)
}

// cc runs the gc-toolchain C compiler in a directory on a C file
// to produce an output file.
func (b *builder) cc(p *Package, objdir, ofile, cfile string) error {
	inc := filepath.Join(b.goroot, "pkg", fmt.Sprintf("%s_%s", b.goos, b.goarch))
	cfile = mkAbs(p.Dir, cfile)
	return b.run(p.Dir, p.ImportPath, b.arch+"c", "-FVw", "-I", objdir, "-I", inc, "-o", ofile, "-DGOOS_"+b.goos, "-DGOARCH_"+b.goarch, cfile)
}

// gcc runs the gcc C compiler to create an object from a single C file.
func (b *builder) gcc(p *Package, out string, flags []string, cfile string) error {
	cfile = mkAbs(p.Dir, cfile)
	return b.run(p.Dir, p.ImportPath, b.gccCmd(p.Dir, flags, "-o", out, "-c", cfile)...)
}

// gccld runs the gcc linker to create an executable from a set of object files
func (b *builder) gccld(p *Package, out string, flags []string, obj []string) error {
	return b.run(p.Dir, p.ImportPath, append(b.gccCmd(p.Dir, flags, "-o", out), obj...)...)
}

// gccCmd returns a gcc command line ending with args
func (b *builder) gccCmd(objdir string, flags []string, args ...string) []string {
	// TODO: HOST_CC?
	a := []string{"gcc", "-I", objdir, "-g", "-fPIC", "-O2"}
	switch b.arch {
	case "8":
		a = append(a, "-m32")
	case "6":
		a = append(a, "-m64")
	}
	a = append(a, flags...)
	return append(a, args...)
}

var cgoRe = regexp.MustCompile(`[/\\:]`)

func (b *builder) cgo(p *Package, obj string, gccfiles []string) (outGo, outObj []string, err error) {
	// cgo
	// TODO: CGOPKGPATH, CGO_FLAGS?
	gofiles := []string{obj + "_cgo_gotypes.go"}
	cfiles := []string{"_cgo_main.c", "_cgo_export.c"}
	for _, fn := range p.CgoFiles {
		f := cgoRe.ReplaceAllString(fn[:len(fn)-2], "_")
		gofiles = append(gofiles, obj+f+"cgo1.go")
		cfiles = append(cfiles, f+"cgo2.c")
	}
	defunC := obj + "_cgo_defun.c"
	// TODO: make cgo not depend on $GOARCH?
	// TODO: make cgo write to obj
	cgoArgs := []string{"cgo", "-objdir", obj}
	if p.Standard && p.ImportPath == "runtime/cgo" {
		cgoArgs = append(cgoArgs, "-import_runtime_cgo=false")
	}
	cgoArgs = append(cgoArgs, "--")
	cgoArgs = append(cgoArgs, p.CgoFiles...)
	if err := b.run(p.Dir, p.ImportPath, cgoArgs...); err != nil {
		return nil, nil, err
	}
	outGo = append(outGo, gofiles...)

	// cc _cgo_defun.c
	defunObj := obj + "_cgo_defun." + b.arch
	if err := b.cc(p, obj, defunObj, defunC); err != nil {
		return nil, nil, err
	}
	outObj = append(outObj, defunObj)

	// gcc
	var linkobj []string
	for _, cfile := range cfiles {
		ofile := obj + cfile[:len(cfile)-1] + "o"
		if err := b.gcc(p, ofile, p.info.CgoCFLAGS, obj+cfile); err != nil {
			return nil, nil, err
		}
		linkobj = append(linkobj, ofile)
		if !strings.HasSuffix(ofile, "_cgo_main.o") {
			outObj = append(outObj, ofile)
		}
	}
	for _, file := range gccfiles {
		ofile := obj + cgoRe.ReplaceAllString(file[:len(file)-1], "_") + "o"
		if err := b.gcc(p, ofile, p.info.CgoCFLAGS, file); err != nil {
			return nil, nil, err
		}
		linkobj = append(linkobj, ofile)
		outObj = append(outObj, ofile)
	}
	dynobj := obj + "_cgo_.o"
	if err := b.gccld(p, dynobj, p.info.CgoLDFLAGS, linkobj); err != nil {
		return nil, nil, err
	}

	// cgo -dynimport
	importC := obj + "_cgo_import.c"
	if err := b.run(p.Dir, p.ImportPath, "cgo", "-objdir", obj, "-dynimport", dynobj, "-dynout", importC); err != nil {
		return nil, nil, err
	}

	// cc _cgo_import.ARCH
	importObj := obj + "_cgo_import." + b.arch
	if err := b.cc(p, obj, importObj, importC); err != nil {
		return nil, nil, err
	}
	outObj = append(outObj, importObj)

	return outGo, outObj, nil
}
