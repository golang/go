// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gobuild

import (
	"flag";
	"fmt";
	"io";
	"os";
	"path";
	"sort";
	"strings";
	"template";
	"unicode";
	"utf8";

	"./gobuild";
)

type Pkg struct

type File struct {
	Name string;
	Pkg *Pkg;
	Imports []string;
	Deps []*Pkg;
	Phase int;
}

type Pkg struct {
	Name string;
	Path string;
	Files []*File;
}

type ArCmd struct {
	Pkg *Pkg;
	Files []*File;
}

type Phase struct {
	Phase int;
	ArCmds []*ArCmd;
}

type Info struct {
	Args []string;
	Char string;
	Dir string;
	ObjDir string;
	Pkgmap map[string] *Pkg;
	Packages []*Pkg;
	Files map[string] *File;
	Imports map[string] bool;
	Phases []*Phase;
	MaxPhase int;
}

var verbose = flag.Bool("v", false, "verbose mode")
var writeMakefile = flag.Bool("m", false, "write Makefile to standard output")

func PushPkg(vp *[]*Pkg, p *Pkg) {
	v := *vp;
	n := len(v);
	if n >= cap(v) {
		m := 2*n + 10;
		a := make([]*Pkg, n, m);
		for i := range v {
			a[i] = v[i];
		}
		v = a;
	}
	v = v[0:n+1];
	v[n] = p;
	*vp = v;
}

func PushFile(vp *[]*File, p *File) {
	v := *vp;
	n := len(v);
	if n >= cap(v) {
		m := 2*n + 10;
		a := make([]*File, n, m);
		for i := range v {
			a[i] = v[i];
		}
		v = a;
	}
	v = v[0:n+1];
	v[n] = p;
	*vp = v;
}

// For sorting Files
type FileArray []*File

func (a FileArray) Len() int {
	return len(a)
}

func (a FileArray) Less(i, j int) bool {
	return a[i].Name < a[j].Name
}

func (a FileArray) Swap(i, j int) {
	a[i], a[j] = a[j], a[i]
}

// If current directory is under $GOROOT/src/pkg, return the
// path relative to there.  Otherwise return "".
func PkgDir() string {
	goroot := os.Getenv("GOROOT");
	if goroot == "" {
		return ""
	}
	srcroot := path.Clean(goroot + "/src/pkg/");
	pwd := os.Getenv("PWD");	// TODO(rsc): real pwd
	if pwd == "" {
		return ""
	}
	if pwd == srcroot {
		return ""
	}
	n := len(srcroot);
	if len(pwd) < n || pwd[n] != '/' || pwd[0:n] != srcroot {
		return ""
	}

	dir := pwd[n+1:len(pwd)];
	return dir;
}

func ScanFiles(filenames []string) *Info {
	// Build list of imports, local packages, and files.
	// Exclude *_test.go and anything in package main.
	// TODO(rsc): Build a binary from package main?

	z := new(Info);
	z.Args = os.Args;
	z.Dir = PkgDir();
	z.Char = theChar;	// for template
	z.ObjDir = ObjDir;	// for template
	z.Pkgmap = make(map[string] *Pkg);
	z.Files = make(map[string] *File);
	z.Imports = make(map[string] bool);

	// Read Go files to find out packages and imports.
	var pkg *Pkg;
	for _, filename := range filenames {
		if strings.Index(filename, "_test.") >= 0 {
			continue;
		}
		f := new(File);
		f.Name = filename;
		if path.Ext(filename) == ".go" {
			rune, _ := utf8.DecodeRuneInString(filename);
			if rune != '_' && !unicode.IsLetter(rune) && !unicode.IsDecimalDigit(rune) {
				// Ignore files with funny leading letters,
				// to avoid editor files like .foo.go and ~foo.go.
				continue;
			}

			pkgname, imp, err := PackageImports(filename);
			if err != nil {
				fatal("parsing %s: %s", filename, err);
			}
			if pkgname == "main" {
				continue;
			}

			path := pkgname;
			var ok bool;
			pkg, ok = z.Pkgmap[path];
			if !ok {
				pkg = new(Pkg);
				pkg.Name = pkgname;
				pkg.Path = path;
				z.Pkgmap[path] = pkg;
				PushPkg(&z.Packages, pkg);
			}
			f.Pkg = pkg;
			f.Imports = imp;
			for _, name := range imp {
				z.Imports[name] = true;
			}
			PushFile(&pkg.Files, f);
		}
		z.Files[filename] = f;
	}

	// Loop through files again, filling in more info.
	for _, f := range z.Files {
		if f.Pkg == nil {
			// non-Go file: fill in package name.
			// Must only be a single package in this directory.
			if len(z.Pkgmap) != 1 {
				fatal("cannot determine package for %s", f.Name);
			}
			f.Pkg = pkg;
		}

		// Go file: record dependencies on other packages in this directory.
		for _, imp := range f.Imports {
			pkg, ok := z.Pkgmap[imp];
			if ok && pkg != f.Pkg {
				PushPkg(&f.Deps, pkg);
			}
		}
	}

	// Update destination directory.
	// If destination directory has same
	// name as package name, cut it off.
	dir, name := path.Split(z.Dir);
	if len(z.Packages) == 1 && z.Packages[0].Name == name {
		z.Dir = dir;
	}

	return z;
}

func PackageObj(pkg string) string {
	return pkg + ".a"
}

func (z *Info) Build() {
	// Create empty object directory tree.
	RemoveAll(ObjDir);
	obj := path.Join(ObjDir, z.Dir) + "/";
	MkdirAll(obj);

	// Create empty archives.
	for pkgname := range z.Pkgmap {
		ar := obj + PackageObj(pkgname);
		os.Remove(ar);
		Archive(ar, nil);
	}

	// Compile by repeated passes: build as many .6 as possible,
	// put them in their archives, and repeat.
	var pending, fail, success []*File;
	for _, file := range z.Files {
		PushFile(&pending, file);
	}
	sort.Sort(FileArray(pending));

	var arfiles []string;
	z.Phases = make([]*Phase, 0, len(z.Files));

	for phase := 1; len(pending) > 0; phase++ {
		// Run what we can.
		fail = fail[0:0];
		success = success[0:0];
		for _, f := range pending {
			if !Build(Compiler(f.Name), f.Name, 0) {
				PushFile(&fail, f);
			} else {
				if *verbose {
					fmt.Fprint(os.Stderr, f.Name, " ");
				}
				PushFile(&success, f);
			}
		}
		if len(success) == 0 {
			// Nothing ran; give up.
			for _, f := range fail {
				Build(Compiler(f.Name), f.Name, ShowErrors | ForceDisplay);
			}
			fatal("stalemate");
		}
		if *verbose {
			fmt.Fprint(os.Stderr, "\n");
		}

		// Record phase data.
		p := new(Phase);
		p.ArCmds = make([]*ArCmd, 0, len(z.Pkgmap));
		p.Phase = phase;
		n := len(z.Phases);
		z.Phases = z.Phases[0:n+1];
		z.Phases[n] = p;

		// Update archives.
		for _, pkg := range z.Pkgmap {
			arfiles = arfiles[0:0];
			var files []*File;
			for _, f := range success {
				if f.Pkg == pkg {
					PushString(&arfiles, Object(f.Name, theChar));
					PushFile(&files, f);
				}
				f.Phase = phase;
			}
			if len(arfiles) > 0 {
				Archive(obj + pkg.Name + ".a", arfiles);

				n := len(p.ArCmds);
				p.ArCmds = p.ArCmds[0:n+1];
				p.ArCmds[n] = &ArCmd{pkg, files};
			}
			for _, filename := range arfiles {
				os.Remove(filename);
			}
		}
		pending, fail = fail, pending;

	}
}

func (z *Info) Clean() {
	RemoveAll(ObjDir);
	for pkgname := range z.Pkgmap {
		os.Remove(PackageObj(pkgname));
	}
}

func Main() {
	flag.Parse();

	filenames := flag.Args();
	if len(filenames) == 0 {
		var err os.Error;
		filenames, err= SourceFiles(".");
		if err != nil {
			fatal("reading .: %s", err.String());
		}
	}

	state := ScanFiles(filenames);
	state.Build();
	if *writeMakefile {
		t, err := template.Parse(makefileTemplate, makefileMap);
		if err != nil {
			fatal("template.Parse: %s", err.String());
		}
		err = t.Execute(state, os.Stdout);
		if err != nil {
			fatal("template.Expand: %s", err.String());
		}
	}
}

