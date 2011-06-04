// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package build provides tools for building Go packages.
package build

import (
	"exec"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

func (d *DirInfo) Build(targ string) ([]*Cmd, os.Error) {
	b := &build{obj: "_obj/"}

	goarch := runtime.GOARCH
	if g := os.Getenv("GOARCH"); g != "" {
		goarch = g
	}
	var err os.Error
	b.arch, err = ArchChar(goarch)
	if err != nil {
		return nil, err
	}

	var gofiles = d.GoFiles // .go files to be built with gc
	var ofiles []string     // *.GOARCH files to be linked or packed

	// make build directory
	b.mkdir(b.obj)

	// cgo
	if len(d.CgoFiles) > 0 {
		outGo, outObj := b.cgo(d.CgoFiles)
		gofiles = append(gofiles, outGo...)
		ofiles = append(ofiles, outObj...)
	}

	// compile
	if len(gofiles) > 0 {
		ofile := b.obj + "_go_." + b.arch
		b.gc(ofile, gofiles...)
		ofiles = append(ofiles, ofile)
	}

	// assemble
	for _, sfile := range d.SFiles {
		ofile := b.obj + sfile[:len(sfile)-1] + b.arch
		b.asm(ofile, sfile)
		ofiles = append(ofiles, ofile)
	}

	if len(ofiles) == 0 {
		return nil, os.NewError("make: no object files to build")
	}

	if d.IsCommand() {
		b.ld(targ, ofiles...)
	} else {
		b.gopack(targ, ofiles...)
	}

	return b.cmds, nil
}

type Cmd struct {
	Args   []string // command-line
	Stdout string   // write standard output to this file, "" is passthrough
	Input  []string // file paths (dependencies)
	Output []string // file paths
}

func (c *Cmd) String() string {
	return strings.Join(c.Args, " ")
}

func (c *Cmd) Run(dir string) os.Error {
	cmd := exec.Command(c.Args[0], c.Args[1:]...)
	cmd.Dir = dir
	if c.Stdout != "" {
		f, err := os.Create(filepath.Join(dir, c.Stdout))
		if err != nil {
			return err
		}
		defer f.Close()
		cmd.Stdout = f
	}
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("command %q: %v", c, err)
	}
	return nil
}

func (c *Cmd) Clean(dir string) (err os.Error) {
	for _, fn := range c.Output {
		if e := os.RemoveAll(fn); err == nil {
			err = e
		}
	}
	return
}

// ArchChar returns the architecture character for the given goarch.
// For example, ArchChar("amd64") returns "6".
func ArchChar(goarch string) (string, os.Error) {
	switch goarch {
	case "386":
		return "8", nil
	case "amd64":
		return "6", nil
	case "arm":
		return "5", nil
	}
	return "", os.NewError("unsupported GOARCH " + goarch)
}

type build struct {
	cmds []*Cmd
	obj  string
	arch string
}

func (b *build) add(c Cmd) {
	b.cmds = append(b.cmds, &c)
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
		Args:   gccArgs(b.arch, "-o", ofile, "-c", cfile),
		Input:  []string{cfile},
		Output: []string{ofile},
	})
}

func (b *build) gccLink(ofile string, ofiles ...string) {
	b.add(Cmd{
		Args:   append(gccArgs(b.arch, "-o", ofile), ofiles...),
		Input:  ofiles,
		Output: []string{ofile},
	})
}

func gccArgs(arch string, args ...string) []string {
	// TODO(adg): HOST_CC
	m := "-m32"
	if arch == "6" {
		m = "-m64"
	}
	return append([]string{"gcc", m, "-I", ".", "-g", "-fPIC", "-O2"}, args...)
}

func (b *build) cgo(cgofiles []string) (outGo, outObj []string) {
	// cgo
	// TODO(adg): CGOPKGPATH
	// TODO(adg): CGO_FLAGS
	gofiles := []string{b.obj + "_cgo_gotypes.go"}
	cfiles := []string{b.obj + "_cgo_main.c", b.obj + "_cgo_export.c"}
	for _, fn := range cgofiles {
		f := b.obj + fn[:len(fn)-2]
		gofiles = append(gofiles, f+"cgo1.go")
		cfiles = append(cfiles, f+"cgo2.c")
	}
	defunC := b.obj + "_cgo_defun.c"
	output := append([]string{defunC}, gofiles...)
	output = append(output, cfiles...)
	b.add(Cmd{
		Args:   append([]string{"cgo", "--"}, cgofiles...),
		Input:  cgofiles,
		Output: output,
	})
	outGo = append(outGo, gofiles...)

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
		}
	}
	dynObj := b.obj + "_cgo1_.o"
	b.gccLink(dynObj, linkobj...)

	// cgo -dynimport
	importC := b.obj + "_cgo_import.c"
	b.add(Cmd{
		Args:   []string{"cgo", "-dynimport", dynObj},
		Stdout: importC,
		Input:  []string{dynObj},
		Output: []string{importC},
	})

	// cc _cgo_import.ARCH
	importObj := b.obj + "_cgo_import." + b.arch
	b.cc(importObj, importC)
	outObj = append(outObj, importObj)

	return
}
