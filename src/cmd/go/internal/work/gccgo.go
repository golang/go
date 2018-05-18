// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package work

import (
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/load"
	"cmd/go/internal/str"
)

// The Gccgo toolchain.

type gccgoToolchain struct{}

var GccgoName, GccgoBin string
var gccgoErr error

func init() {
	GccgoName = os.Getenv("GCCGO")
	if GccgoName == "" {
		GccgoName = "gccgo"
	}
	GccgoBin, gccgoErr = exec.LookPath(GccgoName)
}

func (gccgoToolchain) compiler() string {
	checkGccgoBin()
	return GccgoBin
}

func (gccgoToolchain) linker() string {
	checkGccgoBin()
	return GccgoBin
}

func checkGccgoBin() {
	if gccgoErr == nil {
		return
	}
	fmt.Fprintf(os.Stderr, "cmd/go: gccgo: %s\n", gccgoErr)
	os.Exit(2)
}

func (tools gccgoToolchain) gc(b *Builder, a *Action, archive string, importcfg []byte, asmhdr bool, gofiles []string) (ofile string, output []byte, err error) {
	p := a.Package
	objdir := a.Objdir
	out := "_go_.o"
	ofile = objdir + out
	gcargs := []string{"-g"}
	gcargs = append(gcargs, b.gccArchArgs()...)
	if pkgpath := gccgoPkgpath(p); pkgpath != "" {
		gcargs = append(gcargs, "-fgo-pkgpath="+pkgpath)
	}
	if p.Internal.LocalPrefix != "" {
		gcargs = append(gcargs, "-fgo-relative-import-path="+p.Internal.LocalPrefix)
	}

	args := str.StringList(tools.compiler(), "-c", gcargs, "-o", ofile, forcedGccgoflags)
	if importcfg != nil {
		if b.gccSupportsFlag(args[:1], "-fgo-importcfg=/dev/null") {
			if err := b.writeFile(objdir+"importcfg", importcfg); err != nil {
				return "", nil, err
			}
			args = append(args, "-fgo-importcfg="+objdir+"importcfg")
		} else {
			root := objdir + "_importcfgroot_"
			if err := buildImportcfgSymlinks(b, root, importcfg); err != nil {
				return "", nil, err
			}
			args = append(args, "-I", root)
		}
	}
	args = append(args, a.Package.Internal.Gccgoflags...)
	for _, f := range gofiles {
		args = append(args, mkAbs(p.Dir, f))
	}

	output, err = b.runOut(p.Dir, nil, args)
	return ofile, output, err
}

// buildImportcfgSymlinks builds in root a tree of symlinks
// implementing the directives from importcfg.
// This serves as a temporary transition mechanism until
// we can depend on gccgo reading an importcfg directly.
// (The Go 1.9 and later gc compilers already do.)
func buildImportcfgSymlinks(b *Builder, root string, importcfg []byte) error {
	for lineNum, line := range strings.Split(string(importcfg), "\n") {
		lineNum++ // 1-based
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		var verb, args string
		if i := strings.Index(line, " "); i < 0 {
			verb = line
		} else {
			verb, args = line[:i], strings.TrimSpace(line[i+1:])
		}
		var before, after string
		if i := strings.Index(args, "="); i >= 0 {
			before, after = args[:i], args[i+1:]
		}
		switch verb {
		default:
			base.Fatalf("importcfg:%d: unknown directive %q", lineNum, verb)
		case "packagefile":
			if before == "" || after == "" {
				return fmt.Errorf(`importcfg:%d: invalid packagefile: syntax is "packagefile path=filename": %s`, lineNum, line)
			}
			archive := gccgoArchive(root, before)
			if err := b.Mkdir(filepath.Dir(archive)); err != nil {
				return err
			}
			if err := b.Symlink(after, archive); err != nil {
				return err
			}
		case "importmap":
			if before == "" || after == "" {
				return fmt.Errorf(`importcfg:%d: invalid importmap: syntax is "importmap old=new": %s`, lineNum, line)
			}
			beforeA := gccgoArchive(root, before)
			afterA := gccgoArchive(root, after)
			if err := b.Mkdir(filepath.Dir(beforeA)); err != nil {
				return err
			}
			if err := b.Mkdir(filepath.Dir(afterA)); err != nil {
				return err
			}
			if err := b.Symlink(afterA, beforeA); err != nil {
				return err
			}
		case "packageshlib":
			return fmt.Errorf("gccgo -importcfg does not support shared libraries")
		}
	}
	return nil
}

func (tools gccgoToolchain) asm(b *Builder, a *Action, sfiles []string) ([]string, error) {
	p := a.Package
	var ofiles []string
	for _, sfile := range sfiles {
		base := filepath.Base(sfile)
		ofile := a.Objdir + base[:len(base)-len(".s")] + ".o"
		ofiles = append(ofiles, ofile)
		sfile = mkAbs(p.Dir, sfile)
		defs := []string{"-D", "GOOS_" + cfg.Goos, "-D", "GOARCH_" + cfg.Goarch}
		if pkgpath := gccgoCleanPkgpath(p); pkgpath != "" {
			defs = append(defs, `-D`, `GOPKGPATH=`+pkgpath)
		}
		defs = tools.maybePIC(defs)
		defs = append(defs, b.gccArchArgs()...)
		err := b.run(a, p.Dir, p.ImportPath, nil, tools.compiler(), "-xassembler-with-cpp", "-I", a.Objdir, "-c", "-o", ofile, defs, sfile)
		if err != nil {
			return nil, err
		}
	}
	return ofiles, nil
}

func gccgoArchive(basedir, imp string) string {
	end := filepath.FromSlash(imp + ".a")
	afile := filepath.Join(basedir, end)
	// add "lib" to the final element
	return filepath.Join(filepath.Dir(afile), "lib"+filepath.Base(afile))
}

func (gccgoToolchain) pack(b *Builder, a *Action, afile string, ofiles []string) error {
	p := a.Package
	objdir := a.Objdir
	var absOfiles []string
	for _, f := range ofiles {
		absOfiles = append(absOfiles, mkAbs(objdir, f))
	}
	return b.run(a, p.Dir, p.ImportPath, nil, "ar", "rc", mkAbs(objdir, afile), absOfiles)
}

func (tools gccgoToolchain) link(b *Builder, root *Action, out, importcfg string, allactions []*Action, buildmode, desc string) error {
	// gccgo needs explicit linking with all package dependencies,
	// and all LDFLAGS from cgo dependencies.
	afiles := []string{}
	shlibs := []string{}
	ldflags := b.gccArchArgs()
	cgoldflags := []string{}
	usesCgo := false
	cxx := false
	objc := false
	fortran := false
	if root.Package != nil {
		cxx = len(root.Package.CXXFiles) > 0 || len(root.Package.SwigCXXFiles) > 0
		objc = len(root.Package.MFiles) > 0
		fortran = len(root.Package.FFiles) > 0
	}

	readCgoFlags := func(flagsFile string) error {
		flags, err := ioutil.ReadFile(flagsFile)
		if err != nil {
			return err
		}
		const ldflagsPrefix = "_CGO_LDFLAGS="
		for _, line := range strings.Split(string(flags), "\n") {
			if strings.HasPrefix(line, ldflagsPrefix) {
				newFlags := strings.Fields(line[len(ldflagsPrefix):])
				for _, flag := range newFlags {
					// Every _cgo_flags file has -g and -O2 in _CGO_LDFLAGS
					// but they don't mean anything to the linker so filter
					// them out.
					if flag != "-g" && !strings.HasPrefix(flag, "-O") {
						cgoldflags = append(cgoldflags, flag)
					}
				}
			}
		}
		return nil
	}

	newID := 0
	readAndRemoveCgoFlags := func(archive string) (string, error) {
		newID++
		newArchive := root.Objdir + fmt.Sprintf("_pkg%d_.a", newID)
		if err := b.copyFile(newArchive, archive, 0666, false); err != nil {
			return "", err
		}
		if cfg.BuildN || cfg.BuildX {
			b.Showcmd("", "ar d %s _cgo_flags", newArchive)
			if cfg.BuildN {
				// TODO(rsc): We could do better about showing the right _cgo_flags even in -n mode.
				// Either the archive is already built and we can read them out,
				// or we're printing commands to build the archive and can
				// forward the _cgo_flags directly to this step.
				return "", nil
			}
		}
		err := b.run(root, root.Objdir, desc, nil, "ar", "x", newArchive, "_cgo_flags")
		if err != nil {
			return "", err
		}
		err = b.run(root, ".", desc, nil, "ar", "d", newArchive, "_cgo_flags")
		if err != nil {
			return "", err
		}
		err = readCgoFlags(filepath.Join(root.Objdir, "_cgo_flags"))
		if err != nil {
			return "", err
		}
		return newArchive, nil
	}

	// If using -linkshared, find the shared library deps.
	haveShlib := make(map[string]bool)
	targetBase := filepath.Base(root.Target)
	if cfg.BuildLinkshared {
		for _, a := range root.Deps {
			p := a.Package
			if p == nil || p.Shlib == "" {
				continue
			}

			// The .a we are linking into this .so
			// will have its Shlib set to this .so.
			// Don't start thinking we want to link
			// this .so into itself.
			base := filepath.Base(p.Shlib)
			if base != targetBase {
				haveShlib[base] = true
			}
		}
	}

	// Arrange the deps into afiles and shlibs.
	addedShlib := make(map[string]bool)
	for _, a := range root.Deps {
		p := a.Package
		if p != nil && p.Shlib != "" && haveShlib[filepath.Base(p.Shlib)] {
			// This is a package linked into a shared
			// library that we will put into shlibs.
			continue
		}

		if haveShlib[filepath.Base(a.Target)] {
			// This is a shared library we want to link againt.
			if !addedShlib[a.Target] {
				shlibs = append(shlibs, a.Target)
				addedShlib[a.Target] = true
			}
			continue
		}

		if p != nil {
			target := a.built
			if p.UsesCgo() || p.UsesSwig() {
				var err error
				target, err = readAndRemoveCgoFlags(target)
				if err != nil {
					continue
				}
			}

			afiles = append(afiles, target)
		}
	}

	for _, a := range allactions {
		// Gather CgoLDFLAGS, but not from standard packages.
		// The go tool can dig up runtime/cgo from GOROOT and
		// think that it should use its CgoLDFLAGS, but gccgo
		// doesn't use runtime/cgo.
		if a.Package == nil {
			continue
		}
		if !a.Package.Standard {
			cgoldflags = append(cgoldflags, a.Package.CgoLDFLAGS...)
		}
		if len(a.Package.CgoFiles) > 0 {
			usesCgo = true
		}
		if a.Package.UsesSwig() {
			usesCgo = true
		}
		if len(a.Package.CXXFiles) > 0 || len(a.Package.SwigCXXFiles) > 0 {
			cxx = true
		}
		if len(a.Package.MFiles) > 0 {
			objc = true
		}
		if len(a.Package.FFiles) > 0 {
			fortran = true
		}
	}

	ldflags = append(ldflags, "-Wl,--whole-archive")
	ldflags = append(ldflags, afiles...)
	ldflags = append(ldflags, "-Wl,--no-whole-archive")

	ldflags = append(ldflags, cgoldflags...)
	ldflags = append(ldflags, envList("CGO_LDFLAGS", "")...)
	if root.Package != nil {
		ldflags = append(ldflags, root.Package.CgoLDFLAGS...)
	}

	ldflags = str.StringList("-Wl,-(", ldflags, "-Wl,-)")

	if root.buildID != "" {
		// On systems that normally use gold or the GNU linker,
		// use the --build-id option to write a GNU build ID note.
		switch cfg.Goos {
		case "android", "dragonfly", "linux", "netbsd":
			ldflags = append(ldflags, fmt.Sprintf("-Wl,--build-id=0x%x", root.buildID))
		}
	}

	for _, shlib := range shlibs {
		ldflags = append(
			ldflags,
			"-L"+filepath.Dir(shlib),
			"-Wl,-rpath="+filepath.Dir(shlib),
			"-l"+strings.TrimSuffix(
				strings.TrimPrefix(filepath.Base(shlib), "lib"),
				".so"))
	}

	var realOut string
	switch buildmode {
	case "exe":
		if usesCgo && cfg.Goos == "linux" {
			ldflags = append(ldflags, "-Wl,-E")
		}

	case "c-archive":
		// Link the Go files into a single .o, and also link
		// in -lgolibbegin.
		//
		// We need to use --whole-archive with -lgolibbegin
		// because it doesn't define any symbols that will
		// cause the contents to be pulled in; it's just
		// initialization code.
		//
		// The user remains responsible for linking against
		// -lgo -lpthread -lm in the final link. We can't use
		// -r to pick them up because we can't combine
		// split-stack and non-split-stack code in a single -r
		// link, and libgo picks up non-split-stack code from
		// libffi.
		ldflags = append(ldflags, "-Wl,-r", "-nostdlib", "-Wl,--whole-archive", "-lgolibbegin", "-Wl,--no-whole-archive")

		if nopie := b.gccNoPie([]string{tools.linker()}); nopie != "" {
			ldflags = append(ldflags, nopie)
		}

		// We are creating an object file, so we don't want a build ID.
		if root.buildID == "" {
			ldflags = b.disableBuildID(ldflags)
		}

		realOut = out
		out = out + ".o"

	case "c-shared":
		ldflags = append(ldflags, "-shared", "-nostdlib", "-Wl,--whole-archive", "-lgolibbegin", "-Wl,--no-whole-archive", "-lgo", "-lgcc_s", "-lgcc", "-lc", "-lgcc")
	case "shared":
		ldflags = append(ldflags, "-zdefs", "-shared", "-nostdlib", "-lgo", "-lgcc_s", "-lgcc", "-lc")

	default:
		base.Fatalf("-buildmode=%s not supported for gccgo", buildmode)
	}

	switch buildmode {
	case "exe", "c-shared":
		if cxx {
			ldflags = append(ldflags, "-lstdc++")
		}
		if objc {
			ldflags = append(ldflags, "-lobjc")
		}
		if fortran {
			fc := os.Getenv("FC")
			if fc == "" {
				fc = "gfortran"
			}
			// support gfortran out of the box and let others pass the correct link options
			// via CGO_LDFLAGS
			if strings.Contains(fc, "gfortran") {
				ldflags = append(ldflags, "-lgfortran")
			}
		}
	}

	if err := b.run(root, ".", desc, nil, tools.linker(), "-o", out, ldflags, forcedGccgoflags, root.Package.Internal.Gccgoflags); err != nil {
		return err
	}

	switch buildmode {
	case "c-archive":
		if err := b.run(root, ".", desc, nil, "ar", "rc", realOut, out); err != nil {
			return err
		}
	}
	return nil
}

func (tools gccgoToolchain) ld(b *Builder, root *Action, out, importcfg, mainpkg string) error {
	return tools.link(b, root, out, importcfg, root.Deps, ldBuildmode, root.Package.ImportPath)
}

func (tools gccgoToolchain) ldShared(b *Builder, root *Action, toplevelactions []*Action, out, importcfg string, allactions []*Action) error {
	return tools.link(b, root, out, importcfg, allactions, "shared", out)
}

func (tools gccgoToolchain) cc(b *Builder, a *Action, ofile, cfile string) error {
	p := a.Package
	inc := filepath.Join(cfg.GOROOT, "pkg", "include")
	cfile = mkAbs(p.Dir, cfile)
	defs := []string{"-D", "GOOS_" + cfg.Goos, "-D", "GOARCH_" + cfg.Goarch}
	defs = append(defs, b.gccArchArgs()...)
	if pkgpath := gccgoCleanPkgpath(p); pkgpath != "" {
		defs = append(defs, `-D`, `GOPKGPATH="`+pkgpath+`"`)
	}
	switch cfg.Goarch {
	case "386", "amd64":
		defs = append(defs, "-fsplit-stack")
	}
	defs = tools.maybePIC(defs)
	return b.run(a, p.Dir, p.ImportPath, nil, envList("CC", cfg.DefaultCC(cfg.Goos, cfg.Goarch)), "-Wall", "-g",
		"-I", a.Objdir, "-I", inc, "-o", ofile, defs, "-c", cfile)
}

// maybePIC adds -fPIC to the list of arguments if needed.
func (tools gccgoToolchain) maybePIC(args []string) []string {
	switch cfg.BuildBuildmode {
	case "c-shared", "shared", "plugin":
		args = append(args, "-fPIC")
	}
	return args
}

func gccgoPkgpath(p *load.Package) string {
	if p.Internal.Build.IsCommand() && !p.Internal.ForceLibrary {
		return ""
	}
	return p.ImportPath
}

func gccgoCleanPkgpath(p *load.Package) string {
	clean := func(r rune) rune {
		switch {
		case 'A' <= r && r <= 'Z', 'a' <= r && r <= 'z',
			'0' <= r && r <= '9':
			return r
		}
		return '_'
	}
	return strings.Map(clean, gccgoPkgpath(p))
}
