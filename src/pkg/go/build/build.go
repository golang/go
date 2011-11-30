// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package build provides tools for building Go packages.
package build

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"time"
)

// Build produces a build Script for the given package.
func Build(tree *Tree, pkg string, info *DirInfo) (*Script, error) {
	s := &Script{}
	b := &build{
		script: s,
		path:   filepath.Join(tree.SrcDir(), pkg),
	}
	b.obj = b.abs("_obj") + string(filepath.Separator)

	b.goarch = runtime.GOARCH
	if g := os.Getenv("GOARCH"); g != "" {
		b.goarch = g
	}
	var err error
	b.arch, err = ArchChar(b.goarch)
	if err != nil {
		return nil, err
	}

	// add import object files to list of Inputs
	for _, pkg := range info.Imports {
		t, p, err := FindTree(pkg)
		if err != nil && err != ErrNotFound {
			// FindTree should always be able to suggest an import
			// path and tree. The path must be malformed
			// (for example, an absolute or relative path).
			return nil, errors.New("build: invalid import: " + pkg)
		}
		s.addInput(filepath.Join(t.PkgDir(), p+".a"))
	}

	// .go files to be built with gc
	gofiles := b.abss(info.GoFiles...)
	s.addInput(gofiles...)

	var ofiles []string // object files to be linked or packed

	// make build directory
	b.mkdir(b.obj)
	s.addIntermediate(b.obj)

	// cgo
	if len(info.CgoFiles) > 0 {
		cgoFiles := b.abss(info.CgoFiles...)
		s.addInput(cgoFiles...)
		cgoCFiles := b.abss(info.CFiles...)
		s.addInput(cgoCFiles...)
		outGo, outObj := b.cgo(cgoFiles, cgoCFiles)
		gofiles = append(gofiles, outGo...)
		ofiles = append(ofiles, outObj...)
		s.addIntermediate(outGo...)
		s.addIntermediate(outObj...)
	}

	// compile
	if len(gofiles) > 0 {
		ofile := b.obj + "_go_." + b.arch
		b.gc(ofile, gofiles...)
		ofiles = append(ofiles, ofile)
		s.addIntermediate(ofile)
	}

	// assemble
	for _, sfile := range info.SFiles {
		ofile := b.obj + sfile[:len(sfile)-1] + b.arch
		sfile = b.abs(sfile)
		s.addInput(sfile)
		b.asm(ofile, sfile)
		ofiles = append(ofiles, ofile)
		s.addIntermediate(ofile)
	}

	if len(ofiles) == 0 {
		return nil, errors.New("make: no object files to build")
	}

	// choose target file
	var targ string
	if info.IsCommand() {
		// use the last part of the import path as binary name
		_, bin := filepath.Split(pkg)
		if runtime.GOOS == "windows" {
			bin += ".exe"
		}
		targ = filepath.Join(tree.BinDir(), bin)
	} else {
		targ = filepath.Join(tree.PkgDir(), pkg+".a")
	}

	// make target directory
	targDir, _ := filepath.Split(targ)
	b.mkdir(targDir)

	// link binary or pack object
	if info.IsCommand() {
		b.ld(targ, ofiles...)
	} else {
		b.gopack(targ, ofiles...)
	}
	s.Output = append(s.Output, targ)

	return b.script, nil
}

// A Script describes the build process for a Go package.
// The Input, Intermediate, and Output fields are lists of absolute paths.
type Script struct {
	Cmd          []*Cmd
	Input        []string
	Intermediate []string
	Output       []string
}

func (s *Script) addInput(file ...string) {
	s.Input = append(s.Input, file...)
}

func (s *Script) addIntermediate(file ...string) {
	s.Intermediate = append(s.Intermediate, file...)
}

// Run runs the Script's Cmds in order.
func (s *Script) Run() error {
	for _, c := range s.Cmd {
		if err := c.Run(); err != nil {
			return err
		}
	}
	return nil
}

// Stale returns true if the build's inputs are newer than its outputs.
func (s *Script) Stale() bool {
	var latest time.Time
	// get latest mtime of outputs
	for _, file := range s.Output {
		fi, err := os.Stat(file)
		if err != nil {
			// any error reading output files means stale
			return true
		}
		if mtime := fi.ModTime(); mtime.After(latest) {
			latest = mtime
		}
	}
	for _, file := range s.Input {
		fi, err := os.Stat(file)
		if err != nil || fi.ModTime().After(latest) {
			// any error reading input files means stale
			// (attempt to rebuild to figure out why)
			return true
		}
	}
	return false
}

// Clean removes the Script's Intermediate files.
// It tries to remove every file and returns the first error it encounters.
func (s *Script) Clean() (err error) {
	// Reverse order so that directories get removed after the files they contain.
	for i := len(s.Intermediate) - 1; i >= 0; i-- {
		if e := os.Remove(s.Intermediate[i]); err == nil {
			err = e
		}
	}
	return
}

// Nuke removes the Script's Intermediate and Output files.
// It tries to remove every file and returns the first error it encounters.
func (s *Script) Nuke() (err error) {
	// Reverse order so that directories get removed after the files they contain.
	for i := len(s.Output) - 1; i >= 0; i-- {
		if e := os.Remove(s.Output[i]); err == nil {
			err = e
		}
	}
	if e := s.Clean(); err == nil {
		err = e
	}
	return
}

// A Cmd describes an individual build command.
type Cmd struct {
	Args   []string // command-line
	Stdout string   // write standard output to this file, "" is passthrough
	Dir    string   // working directory
	Env    []string // environment
	Input  []string // file paths (dependencies)
	Output []string // file paths
}

func (c *Cmd) String() string {
	return strings.Join(c.Args, " ")
}

// Run executes the Cmd.
func (c *Cmd) Run() error {
	if c.Args[0] == "mkdir" {
		for _, p := range c.Output {
			if err := os.MkdirAll(p, 0777); err != nil {
				return fmt.Errorf("command %q: %v", c, err)
			}
		}
		return nil
	}
	out := new(bytes.Buffer)
	cmd := exec.Command(c.Args[0], c.Args[1:]...)
	cmd.Dir = c.Dir
	cmd.Env = c.Env
	cmd.Stdout = out
	cmd.Stderr = out
	if c.Stdout != "" {
		f, err := os.Create(c.Stdout)
		if err != nil {
			return err
		}
		defer f.Close()
		cmd.Stdout = f
	}
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("command %q: %v\n%v", c, err, out)
	}
	return nil
}

// ArchChar returns the architecture character for the given goarch.
// For example, ArchChar("amd64") returns "6".
func ArchChar(goarch string) (string, error) {
	switch goarch {
	case "386":
		return "8", nil
	case "amd64":
		return "6", nil
	case "arm":
		return "5", nil
	}
	return "", errors.New("unsupported GOARCH " + goarch)
}

type build struct {
	script *Script
	path   string
	obj    string
	goarch string
	arch   string
}

func (b *build) abs(file string) string {
	if filepath.IsAbs(file) {
		return file
	}
	return filepath.Join(b.path, file)
}

func (b *build) abss(file ...string) []string {
	s := make([]string, len(file))
	for i, f := range file {
		s[i] = b.abs(f)
	}
	return s
}

func (b *build) add(c Cmd) {
	b.script.Cmd = append(b.script.Cmd, &c)
}

func (b *build) mkdir(name string) {
	b.add(Cmd{
		Args:   []string{"mkdir", "-p", name},
		Output: []string{name},
	})
}

func (b *build) gc(ofile string, gofiles ...string) {
	gc := b.arch + "g"
	args := append([]string{gc, "-o", ofile}, gcImportArgs...)
	args = append(args, gofiles...)
	b.add(Cmd{
		Args:   args,
		Input:  gofiles,
		Output: []string{ofile},
	})
}

func (b *build) asm(ofile string, sfile string) {
	asm := b.arch + "a"
	b.add(Cmd{
		Args:   []string{asm, "-o", ofile, sfile},
		Input:  []string{sfile},
		Output: []string{ofile},
	})
}

func (b *build) ld(targ string, ofiles ...string) {
	ld := b.arch + "l"
	args := append([]string{ld, "-o", targ}, ldImportArgs...)
	args = append(args, ofiles...)
	b.add(Cmd{
		Args:   args,
		Input:  ofiles,
		Output: []string{targ},
	})
}

func (b *build) gopack(targ string, ofiles ...string) {
	b.add(Cmd{
		Args:   append([]string{"gopack", "grc", targ}, ofiles...),
		Input:  ofiles,
		Output: []string{targ},
	})
}

func (b *build) cc(ofile string, cfiles ...string) {
	cc := b.arch + "c"
	dir := fmt.Sprintf("%s_%s", runtime.GOOS, runtime.GOARCH)
	inc := filepath.Join(runtime.GOROOT(), "pkg", dir)
	args := []string{cc, "-FVw", "-I", inc, "-o", ofile}
	b.add(Cmd{
		Args:   append(args, cfiles...),
		Input:  cfiles,
		Output: []string{ofile},
	})
}

func (b *build) gccCompile(ofile, cfile string) {
	b.add(Cmd{
		Args:   b.gccArgs("-o", ofile, "-c", cfile),
		Input:  []string{cfile},
		Output: []string{ofile},
	})
}

func (b *build) gccLink(ofile string, ofiles ...string) {
	b.add(Cmd{
		Args:   append(b.gccArgs("-o", ofile), ofiles...),
		Input:  ofiles,
		Output: []string{ofile},
	})
}

func (b *build) gccArgs(args ...string) []string {
	// TODO(adg): HOST_CC
	a := []string{"gcc", "-I", b.path, "-g", "-fPIC", "-O2"}
	switch b.arch {
	case "8":
		a = append(a, "-m32")
	case "6":
		a = append(a, "-m64")
	}
	return append(a, args...)
}

var cgoRe = regexp.MustCompile(`[/\\:]`)

func (b *build) cgo(cgofiles, cgocfiles []string) (outGo, outObj []string) {
	// cgo
	// TODO(adg): CGOPKGPATH
	// TODO(adg): CGO_FLAGS
	gofiles := []string{b.obj + "_cgo_gotypes.go"}
	cfiles := []string{b.obj + "_cgo_main.c", b.obj + "_cgo_export.c"}
	for _, fn := range cgofiles {
		f := b.obj + cgoRe.ReplaceAllString(fn[:len(fn)-2], "_")
		gofiles = append(gofiles, f+"cgo1.go")
		cfiles = append(cfiles, f+"cgo2.c")
	}
	defunC := b.obj + "_cgo_defun.c"
	output := append([]string{defunC}, cfiles...)
	output = append(output, gofiles...)
	b.add(Cmd{
		Args:   append([]string{"cgo", "--"}, cgofiles...),
		Dir:    b.path,
		Env:    append(os.Environ(), "GOARCH="+b.goarch),
		Input:  cgofiles,
		Output: output,
	})
	outGo = append(outGo, gofiles...)
	exportH := filepath.Join(b.path, "_cgo_export.h")
	b.script.addIntermediate(defunC, exportH, b.obj+"_cgo_flags")
	b.script.addIntermediate(cfiles...)

	// cc _cgo_defun.c
	defunObj := b.obj + "_cgo_defun." + b.arch
	b.cc(defunObj, defunC)
	outObj = append(outObj, defunObj)

	// gcc
	linkobj := make([]string, 0, len(cfiles))
	for _, cfile := range cfiles {
		ofile := cfile[:len(cfile)-1] + "o"
		b.gccCompile(ofile, cfile)
		linkobj = append(linkobj, ofile)
		if !strings.HasSuffix(ofile, "_cgo_main.o") {
			outObj = append(outObj, ofile)
		} else {
			b.script.addIntermediate(ofile)
		}
	}
	for _, cfile := range cgocfiles {
		ofile := b.obj + cgoRe.ReplaceAllString(cfile[:len(cfile)-1], "_") + "o"
		b.gccCompile(ofile, cfile)
		linkobj = append(linkobj, ofile)
		outObj = append(outObj, ofile)
	}
	dynObj := b.obj + "_cgo_.o"
	b.gccLink(dynObj, linkobj...)
	b.script.addIntermediate(dynObj)

	// cgo -dynimport
	importC := b.obj + "_cgo_import.c"
	b.add(Cmd{
		Args:   []string{"cgo", "-dynimport", dynObj},
		Stdout: importC,
		Input:  []string{dynObj},
		Output: []string{importC},
	})
	b.script.addIntermediate(importC)

	// cc _cgo_import.ARCH
	importObj := b.obj + "_cgo_import." + b.arch
	b.cc(importObj, importC)
	outObj = append(outObj, importObj)

	return
}
