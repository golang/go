// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"fmt"
	"go/build"
	"io"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
)

// Break init cycles
func init() {
	cmdBuild.Run = runBuild
	cmdInstall.Run = runInstall
}

var cmdBuild = &Command{
	UsageLine: "build [-a] [-n] [-v] [importpath... | gofiles...]",
	Short:     "compile packages and dependencies",
	Long: `
Build compiles the packages named by the import paths,
along with their dependencies, but it does not install the results.

If the arguments are a list of .go files, build compiles them into
a package object or command executable named for the first
source file.

The -a flag forces rebuilding of packages that are already up-to-date.
The -n flag prints the commands but does not run them.
The -v flag prints the commands.

For more about import paths, see 'go help importpath'.

See also: go install, go get, go clean.
	`,
}

var buildA = cmdBuild.Flag.Bool("a", false, "")
var buildN = cmdBuild.Flag.Bool("n", false, "")
var buildV = cmdBuild.Flag.Bool("v", false, "")

func runBuild(cmd *Command, args []string) {
	var b builder
	b.init(*buildA, *buildN, *buildV)

	if len(args) > 0 && strings.HasSuffix(args[0], ".go") {
		b.do(b.action(modeInstall, modeBuild, goFilesPackage(args, "")))
		return
	}

	a := &action{f: (*builder).nop}
	for _, p := range packages(args) {
		a.deps = append(a.deps, b.action(modeBuild, modeBuild, p))
	}
	b.do(a)
}

var cmdInstall = &Command{
	UsageLine: "install [-a] [-n] [-v] [importpath...]",
	Short:     "compile and install packages and dependencies",
	Long: `
Install compiles and installs the packages named by the import paths,
along with their dependencies.

The -a flag forces reinstallation of packages that are already up-to-date.
The -n flag prints the commands but does not run them.
The -v flag prints the commands.

For more about import paths, see 'go help importpath'.

See also: go build, go get, go clean.
	`,
}

var installA = cmdInstall.Flag.Bool("a", false, "")
var installN = cmdInstall.Flag.Bool("n", false, "")
var installV = cmdInstall.Flag.Bool("v", false, "")

func runInstall(cmd *Command, args []string) {
	var b builder
	b.init(*installA, *installN, *installV)
	a := &action{f: (*builder).nop}
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
	vflag       bool                 // the -v flag
	arch        string               // e.g., "6"
	goroot      string               // the $GOROOT
	goarch      string               // the $GOARCH
	goos        string               // the $GOOS
	actionCache map[cacheKey]*action // a cache of already-constructed actions
}

// An action represents a single action in the action graph.
type action struct {
	f func(*builder, *action) error // the action itself

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

func (b *builder) init(aflag, nflag, vflag bool) {
	var err error
	b.aflag = aflag
	b.nflag = nflag
	b.vflag = vflag
	b.actionCache = make(map[cacheKey]*action)
	b.goroot = runtime.GOROOT()
	b.goarch = build.DefaultContext.GOARCH
	b.goos = build.DefaultContext.GOOS

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
		if vflag {
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

		if !needInstall(p) && !b.aflag {
			// TODO: This is not right if the deps above
			// are not all no-ops too.  If fmt is up to date
			// wrt its own source files,  but strconv has
			// changed, then fmt is not up to date.
			a.f = (*builder).nop
			return a
		}
		if p.Standard {
			switch p.ImportPath {
			case "runtime/cgo":
				// Too complex - can't build.
				a.f = (*builder).nop
				return a
			case "builtin", "unsafe":
				// Fake packages - nothing to build.
				a.f = (*builder).nop
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
	if err := a.f(b, a); err != nil {
		errorf("%s", err)
		a.failed = true
	}
	a.done = true
}

func (b *builder) nop(a *action) error {
	return nil
}

// build is the action for building a single package or command.
func (b *builder) build(a *action) error {
	obj := filepath.Join(b.work, filepath.FromSlash(a.p.ImportPath+"/_obj")) + string(filepath.Separator)
	if a.pkgobj == "" {
		a.pkgobj = filepath.Join(b.work, filepath.FromSlash(a.p.ImportPath+".a"))
	}

	// make build directory
	if err := b.mkdir(obj); err != nil {
		return err
	}

	var objects []string
	var gofiles []string
	gofiles = append(gofiles, a.p.GoFiles...)

	// run cgo
	if len(a.p.CgoFiles) > 0 {
		outGo, outObj, err := b.cgo(a.p.Dir, obj, a.p.info)
		if err != nil {
			return err
		}
		objects = append(objects, outObj...)
		gofiles = append(gofiles, outGo...)
	}

	// prepare Go import path list
	var inc []string
	inc = append(inc, "-I", b.work)
	incMap := map[string]bool{}
	for _, a1 := range a.deps {
		pkgdir := a1.pkgdir
		if pkgdir == build.Path[0].PkgDir() || pkgdir == "" {
			continue
		}
		if !incMap[pkgdir] {
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
		if err := b.gc(a.p.Dir, obj+out, gcargs, inc, gofiles); err != nil {
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

	// in a cgo package, the .c files are compiled with gcc during b.cgo above.
	// in a non-cgo package, the .c files are compiled with 5c/6c/8c.
	// The same convention applies for .s files.
	if len(a.p.CgoFiles) == 0 {
		for _, file := range a.p.CFiles {
			out := file[:len(file)-len(".c")] + "." + b.arch
			if err := b.cc(a.p.Dir, obj+out, file); err != nil {
				return err
			}
			objects = append(objects, out)
		}

		// assemble .s files
		for _, file := range a.p.SFiles {
			out := file[:len(file)-len(".s")] + "." + b.arch
			if err := b.asm(a.p.Dir, obj+out, file); err != nil {
				return err
			}
			objects = append(objects, out)
		}
	}

	// pack into archive
	if err := b.gopack(obj, a.pkgobj, objects); err != nil {
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
		if err := b.ld(a.p.Dir, a.pkgbin, inc, a.pkgobj); err != nil {
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
	if b.nflag || b.vflag {
		b.showcmd("cp %s %s", src, dst)
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

// fmtcmd is like fmt.Sprintf but replaces references to the
// work directory (a temporary directory with a clumsy name)
// with $WORK.
func (b *builder) fmtcmd(format string, args ...interface{}) string {
	s := fmt.Sprintf(format, args...)
	s = strings.Replace(s, b.work, "$WORK", -1)
	return s
}

// showcmd prints the given command to standard output
// for the implementation of -n or -v.
func (b *builder) showcmd(format string, args ...interface{}) {
	fmt.Println(b.fmtcmd(format, args...))
}

// run runs the command given by cmdline in the directory dir.
// If the commnd fails, run prints information about the failure
// and returns a non-nil error.
func (b *builder) run(dir string, cmdline ...string) error {
	if b.nflag || b.vflag {
		b.showcmd("cd %s; %s", dir, strings.Join(cmdline, " "))
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
		fmt.Fprintf(os.Stderr, "# cd %s; %s\n", dir, strings.Join(cmdline, " "))
		fmt.Fprintf(os.Stderr, "%s\n", buf.Bytes())
	}
	return err
}

// mkdir makes the named directory.
func (b *builder) mkdir(dir string) error {
	if b.nflag || b.vflag {
		b.showcmd("mkdir -p %s", dir)
		if b.nflag {
			return nil
		}
	}

	if err := os.MkdirAll(dir, 0777); err != nil {
		return err
	}
	return nil
}

// gc runs the Go compiler in a specific directory on a set of files
// to generate the named output file. 
func (b *builder) gc(dir, ofile string, gcargs, importArgs []string, gofiles []string) error {
	args := []string{b.arch + "g", "-o", ofile}
	args = append(args, gcargs...)
	args = append(args, importArgs...)
	args = append(args, gofiles...)
	return b.run(dir, args...)
}

// asm runs the assembler in a specific directory on a specific file
// to generate the named output file. 
func (b *builder) asm(dir, ofile, sfile string) error {
	return b.run(dir, b.arch+"a", "-o", ofile, "-DGOOS_"+b.goos, "-DGOARCH_"+b.goarch, sfile)
}

// gopack runs the assembler in a specific directory to create
// an archive from a set of object files.
// typically it is run in the object directory.
func (b *builder) gopack(objDir, afile string, ofiles []string) error {
	return b.run(objDir, append([]string{"gopack", "grc", afile}, ofiles...)...)
}

// ld runs the linker to create a package starting at mainpkg.
func (b *builder) ld(dir, out string, importArgs []string, mainpkg string) error {
	return b.run(dir, append(append([]string{b.arch + "l", "-o", out}, importArgs...), mainpkg)...)
}

// cc runs the gc-toolchain C compiler in a directory on a C file
// to produce an output file.
func (b *builder) cc(dir, ofile, cfile string) error {
	inc := filepath.Join(runtime.GOROOT(), "pkg",
		fmt.Sprintf("%s_%s", b.goos, b.goarch))
	return b.run(dir, b.arch+"c", "-FVw", "-I", inc, "-o", ofile, "-DGOOS_"+b.goos, "-DGOARCH_"+b.goarch, cfile)
}

// gcc runs the gcc C compiler to create an object from a single C file.
func (b *builder) gcc(dir, out string, flags []string, cfile string) error {
	return b.run(dir, b.gccCmd(dir, flags, "-o", out, "-c", cfile)...)
}

// gccld runs the gcc linker to create an executable from a set of object files
func (b *builder) gccld(dir, out string, flags []string, obj []string) error {
	return b.run(dir, append(b.gccCmd(dir, flags, "-o", out), obj...)...)
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

func (b *builder) cgo(dir, obj string, info *build.DirInfo) (outGo, outObj []string, err error) {
	// cgo
	// TODO: CGOPKGPATH, CGO_FLAGS?
	gofiles := []string{obj + "_cgo_gotypes.go"}
	cfiles := []string{"_cgo_main.c", "_cgo_export.c"}
	for _, fn := range info.CgoFiles {
		f := cgoRe.ReplaceAllString(fn[:len(fn)-2], "_")
		gofiles = append(gofiles, obj+f+"cgo1.go")
		cfiles = append(cfiles, f+"cgo2.c")
	}
	defunC := obj + "_cgo_defun.c"
	// TODO: make cgo not depend on $GOARCH?
	// TODO: make cgo write to obj
	if err := b.run(dir, append([]string{"cgo", "-objdir", obj, "--"}, info.CgoFiles...)...); err != nil {
		return nil, nil, err
	}
	outGo = append(outGo, gofiles...)

	// cc _cgo_defun.c
	defunObj := obj + "_cgo_defun." + b.arch
	if err := b.cc(dir, defunObj, defunC); err != nil {
		return nil, nil, err
	}
	outObj = append(outObj, defunObj)

	// gcc
	var linkobj []string
	for _, cfile := range cfiles {
		ofile := obj + cfile[:len(cfile)-1] + "o"
		if err := b.gcc(dir, ofile, info.CgoCFLAGS, obj+cfile); err != nil {
			return nil, nil, err
		}
		linkobj = append(linkobj, ofile)
		if !strings.HasSuffix(ofile, "_cgo_main.o") {
			outObj = append(outObj, ofile)
		}
	}
	for _, cfile := range info.CFiles {
		ofile := obj + cgoRe.ReplaceAllString(cfile[:len(cfile)-1], "_") + "o"
		if err := b.gcc(dir, ofile, info.CgoCFLAGS, cfile); err != nil {
			return nil, nil, err
		}
		linkobj = append(linkobj, ofile)
		outObj = append(outObj, ofile)
	}
	dynobj := obj + "_cgo_.o"
	if err := b.gccld(dir, dynobj, info.CgoLDFLAGS, linkobj); err != nil {
		return nil, nil, err
	}

	// cgo -dynimport
	importC := obj + "_cgo_import.c"
	if err := b.run(dir, "cgo", "-objdir", obj, "-dynimport", dynobj, "-dynout", importC); err != nil {
		return nil, nil, err
	}

	// cc _cgo_import.ARCH
	importObj := obj + "_cgo_import." + b.arch
	if err := b.cc(dir, importObj, importC); err != nil {
		return nil, nil, err
	}
	outObj = append(outObj, importObj)

	return outGo, outObj, nil
}
