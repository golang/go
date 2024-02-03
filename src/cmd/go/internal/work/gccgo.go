// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package work

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/fsys"
	"cmd/go/internal/load"
	"cmd/go/internal/str"
	"cmd/internal/pkgpath"
)

// The Gccgo toolchain.

type gccgoToolchain struct{}

var GccgoName, GccgoBin string
var gccgoErr error

func init() {
	GccgoName = cfg.Getenv("GCCGO")
	if GccgoName == "" {
		GccgoName = "gccgo"
	}
	GccgoBin, gccgoErr = cfg.LookPath(GccgoName)
}

func (gccgoToolchain) compiler() string {
	checkGccgoBin()
	return GccgoBin
}

func (gccgoToolchain) linker() string {
	checkGccgoBin()
	return GccgoBin
}

func (gccgoToolchain) ar() []string {
	return envList("AR", "ar")
}

func checkGccgoBin() {
	if gccgoErr == nil {
		return
	}
	fmt.Fprintf(os.Stderr, "cmd/go: gccgo: %s\n", gccgoErr)
	base.SetExitStatus(2)
	base.Exit()
}

func (tools gccgoToolchain) gc(b *Builder, a *Action, archive string, importcfg, embedcfg []byte, symabis string, asmhdr bool, gofiles []string) (ofile string, output []byte, err error) {
	p := a.Package
	sh := b.Shell(a)
	objdir := a.Objdir
	out := "_go_.o"
	ofile = objdir + out
	gcargs := []string{"-g"}
	gcargs = append(gcargs, b.gccArchArgs()...)
	gcargs = append(gcargs, "-fdebug-prefix-map="+b.WorkDir+"=/tmp/go-build")
	gcargs = append(gcargs, "-gno-record-gcc-switches")
	if pkgpath := gccgoPkgpath(p); pkgpath != "" {
		gcargs = append(gcargs, "-fgo-pkgpath="+pkgpath)
	}
	if p.Internal.LocalPrefix != "" {
		gcargs = append(gcargs, "-fgo-relative-import-path="+p.Internal.LocalPrefix)
	}

	args := str.StringList(tools.compiler(), "-c", gcargs, "-o", ofile, forcedGccgoflags)
	if importcfg != nil {
		if b.gccSupportsFlag(args[:1], "-fgo-importcfg=/dev/null") {
			if err := sh.writeFile(objdir+"importcfg", importcfg); err != nil {
				return "", nil, err
			}
			args = append(args, "-fgo-importcfg="+objdir+"importcfg")
		} else {
			root := objdir + "_importcfgroot_"
			if err := buildImportcfgSymlinks(sh, root, importcfg); err != nil {
				return "", nil, err
			}
			args = append(args, "-I", root)
		}
	}
	if embedcfg != nil && b.gccSupportsFlag(args[:1], "-fgo-embedcfg=/dev/null") {
		if err := sh.writeFile(objdir+"embedcfg", embedcfg); err != nil {
			return "", nil, err
		}
		args = append(args, "-fgo-embedcfg="+objdir+"embedcfg")
	}

	if b.gccSupportsFlag(args[:1], "-ffile-prefix-map=a=b") {
		if cfg.BuildTrimpath {
			args = append(args, "-ffile-prefix-map="+base.Cwd()+"=.")
			args = append(args, "-ffile-prefix-map="+b.WorkDir+"=/tmp/go-build")
		}
		if fsys.OverlayFile != "" {
			for _, name := range gofiles {
				absPath := mkAbs(p.Dir, name)
				overlayPath, ok := fsys.OverlayPath(absPath)
				if !ok {
					continue
				}
				toPath := absPath
				// gccgo only applies the last matching rule, so also handle the case where
				// BuildTrimpath is true and the path is relative to base.Cwd().
				if cfg.BuildTrimpath && str.HasFilePathPrefix(toPath, base.Cwd()) {
					toPath = "." + toPath[len(base.Cwd()):]
				}
				args = append(args, "-ffile-prefix-map="+overlayPath+"="+toPath)
			}
		}
	}

	args = append(args, a.Package.Internal.Gccgoflags...)
	for _, f := range gofiles {
		f := mkAbs(p.Dir, f)
		// Overlay files if necessary.
		// See comment on gctoolchain.gc about overlay TODOs
		f, _ = fsys.OverlayPath(f)
		args = append(args, f)
	}

	output, err = sh.runOut(p.Dir, nil, args)
	return ofile, output, err
}

// buildImportcfgSymlinks builds in root a tree of symlinks
// implementing the directives from importcfg.
// This serves as a temporary transition mechanism until
// we can depend on gccgo reading an importcfg directly.
// (The Go 1.9 and later gc compilers already do.)
func buildImportcfgSymlinks(sh *Shell, root string, importcfg []byte) error {
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
		before, after, _ := strings.Cut(args, "=")
		switch verb {
		default:
			base.Fatalf("importcfg:%d: unknown directive %q", lineNum, verb)
		case "packagefile":
			if before == "" || after == "" {
				return fmt.Errorf(`importcfg:%d: invalid packagefile: syntax is "packagefile path=filename": %s`, lineNum, line)
			}
			archive := gccgoArchive(root, before)
			if err := sh.Mkdir(filepath.Dir(archive)); err != nil {
				return err
			}
			if err := sh.Symlink(after, archive); err != nil {
				return err
			}
		case "importmap":
			if before == "" || after == "" {
				return fmt.Errorf(`importcfg:%d: invalid importmap: syntax is "importmap old=new": %s`, lineNum, line)
			}
			beforeA := gccgoArchive(root, before)
			afterA := gccgoArchive(root, after)
			if err := sh.Mkdir(filepath.Dir(beforeA)); err != nil {
				return err
			}
			if err := sh.Mkdir(filepath.Dir(afterA)); err != nil {
				return err
			}
			if err := sh.Symlink(afterA, beforeA); err != nil {
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
		sfile, _ = fsys.OverlayPath(mkAbs(p.Dir, sfile))
		defs := []string{"-D", "GOOS_" + cfg.Goos, "-D", "GOARCH_" + cfg.Goarch}
		if pkgpath := tools.gccgoCleanPkgpath(b, p); pkgpath != "" {
			defs = append(defs, `-D`, `GOPKGPATH=`+pkgpath)
		}
		defs = tools.maybePIC(defs)
		defs = append(defs, b.gccArchArgs()...)
		err := b.Shell(a).run(p.Dir, p.ImportPath, nil, tools.compiler(), "-xassembler-with-cpp", "-I", a.Objdir, "-c", "-o", ofile, defs, sfile)
		if err != nil {
			return nil, err
		}
	}
	return ofiles, nil
}

func (gccgoToolchain) symabis(b *Builder, a *Action, sfiles []string) (string, error) {
	return "", nil
}

func gccgoArchive(basedir, imp string) string {
	end := filepath.FromSlash(imp + ".a")
	afile := filepath.Join(basedir, end)
	// add "lib" to the final element
	return filepath.Join(filepath.Dir(afile), "lib"+filepath.Base(afile))
}

func (tools gccgoToolchain) pack(b *Builder, a *Action, afile string, ofiles []string) error {
	p := a.Package
	sh := b.Shell(a)
	objdir := a.Objdir
	var absOfiles []string
	for _, f := range ofiles {
		absOfiles = append(absOfiles, mkAbs(objdir, f))
	}
	var arArgs []string
	if cfg.Goos == "aix" && cfg.Goarch == "ppc64" {
		// AIX puts both 32-bit and 64-bit objects in the same archive.
		// Tell the AIX "ar" command to only care about 64-bit objects.
		arArgs = []string{"-X64"}
	}
	absAfile := mkAbs(objdir, afile)
	// Try with D modifier first, then without if that fails.
	output, err := sh.runOut(p.Dir, nil, tools.ar(), arArgs, "rcD", absAfile, absOfiles)
	if err != nil {
		return sh.run(p.Dir, p.ImportPath, nil, tools.ar(), arArgs, "rc", absAfile, absOfiles)
	}

	// Show the output if there is any even without errors.
	return sh.reportCmd("", "", output, nil)
}

func (tools gccgoToolchain) link(b *Builder, root *Action, out, importcfg string, allactions []*Action, buildmode, desc string) error {
	sh := b.Shell(root)

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
		flags, err := os.ReadFile(flagsFile)
		if err != nil {
			return err
		}
		const ldflagsPrefix = "_CGO_LDFLAGS="
		for _, line := range strings.Split(string(flags), "\n") {
			if strings.HasPrefix(line, ldflagsPrefix) {
				flag := line[len(ldflagsPrefix):]
				// Every _cgo_flags file has -g and -O2 in _CGO_LDFLAGS
				// but they don't mean anything to the linker so filter
				// them out.
				if flag != "-g" && !strings.HasPrefix(flag, "-O") {
					cgoldflags = append(cgoldflags, flag)
				}
			}
		}
		return nil
	}

	var arArgs []string
	if cfg.Goos == "aix" && cfg.Goarch == "ppc64" {
		// AIX puts both 32-bit and 64-bit objects in the same archive.
		// Tell the AIX "ar" command to only care about 64-bit objects.
		arArgs = []string{"-X64"}
	}

	newID := 0
	readAndRemoveCgoFlags := func(archive string) (string, error) {
		newID++
		newArchive := root.Objdir + fmt.Sprintf("_pkg%d_.a", newID)
		if err := sh.CopyFile(newArchive, archive, 0666, false); err != nil {
			return "", err
		}
		if cfg.BuildN || cfg.BuildX {
			sh.ShowCmd("", "ar d %s _cgo_flags", newArchive)
			if cfg.BuildN {
				// TODO(rsc): We could do better about showing the right _cgo_flags even in -n mode.
				// Either the archive is already built and we can read them out,
				// or we're printing commands to build the archive and can
				// forward the _cgo_flags directly to this step.
				return "", nil
			}
		}
		err := sh.run(root.Objdir, desc, nil, tools.ar(), arArgs, "x", newArchive, "_cgo_flags")
		if err != nil {
			return "", err
		}
		err = sh.run(".", desc, nil, tools.ar(), arArgs, "d", newArchive, "_cgo_flags")
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
			// This is a shared library we want to link against.
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
		if a.Package == nil {
			continue
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

	wholeArchive := []string{"-Wl,--whole-archive"}
	noWholeArchive := []string{"-Wl,--no-whole-archive"}
	if cfg.Goos == "aix" {
		wholeArchive = nil
		noWholeArchive = nil
	}
	ldflags = append(ldflags, wholeArchive...)
	ldflags = append(ldflags, afiles...)
	ldflags = append(ldflags, noWholeArchive...)

	ldflags = append(ldflags, cgoldflags...)
	ldflags = append(ldflags, envList("CGO_LDFLAGS", "")...)
	if cfg.Goos != "aix" {
		ldflags = str.StringList("-Wl,-(", ldflags, "-Wl,-)")
	}

	if root.buildID != "" {
		// On systems that normally use gold or the GNU linker,
		// use the --build-id option to write a GNU build ID note.
		switch cfg.Goos {
		case "android", "dragonfly", "linux", "netbsd":
			ldflags = append(ldflags, fmt.Sprintf("-Wl,--build-id=0x%x", root.buildID))
		}
	}

	var rLibPath string
	if cfg.Goos == "aix" {
		rLibPath = "-Wl,-blibpath="
	} else {
		rLibPath = "-Wl,-rpath="
	}
	for _, shlib := range shlibs {
		ldflags = append(
			ldflags,
			"-L"+filepath.Dir(shlib),
			rLibPath+filepath.Dir(shlib),
			"-l"+strings.TrimSuffix(
				strings.TrimPrefix(filepath.Base(shlib), "lib"),
				".so"))
	}

	var realOut string
	goLibBegin := str.StringList(wholeArchive, "-lgolibbegin", noWholeArchive)
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
		ldflags = append(ldflags, "-Wl,-r", "-nostdlib")
		ldflags = append(ldflags, goLibBegin...)

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
		ldflags = append(ldflags, "-shared", "-nostdlib")
		ldflags = append(ldflags, goLibBegin...)
		ldflags = append(ldflags, "-lgo", "-lgcc_s", "-lgcc", "-lc", "-lgcc")

	case "shared":
		if cfg.Goos != "aix" {
			ldflags = append(ldflags, "-zdefs")
		}
		ldflags = append(ldflags, "-shared", "-nostdlib", "-lgo", "-lgcc_s", "-lgcc", "-lc")

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
			fc := cfg.Getenv("FC")
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

	if err := sh.run(".", desc, nil, tools.linker(), "-o", out, ldflags, forcedGccgoflags, root.Package.Internal.Gccgoflags); err != nil {
		return err
	}

	switch buildmode {
	case "c-archive":
		if err := sh.run(".", desc, nil, tools.ar(), arArgs, "rc", realOut, out); err != nil {
			return err
		}
	}
	return nil
}

func (tools gccgoToolchain) ld(b *Builder, root *Action, targetPath, importcfg, mainpkg string) error {
	return tools.link(b, root, targetPath, importcfg, root.Deps, ldBuildmode, root.Package.ImportPath)
}

func (tools gccgoToolchain) ldShared(b *Builder, root *Action, toplevelactions []*Action, targetPath, importcfg string, allactions []*Action) error {
	return tools.link(b, root, targetPath, importcfg, allactions, "shared", targetPath)
}

func (tools gccgoToolchain) cc(b *Builder, a *Action, ofile, cfile string) error {
	p := a.Package
	inc := filepath.Join(cfg.GOROOT, "pkg", "include")
	cfile = mkAbs(p.Dir, cfile)
	defs := []string{"-D", "GOOS_" + cfg.Goos, "-D", "GOARCH_" + cfg.Goarch}
	defs = append(defs, b.gccArchArgs()...)
	if pkgpath := tools.gccgoCleanPkgpath(b, p); pkgpath != "" {
		defs = append(defs, `-D`, `GOPKGPATH="`+pkgpath+`"`)
	}
	compiler := envList("CC", cfg.DefaultCC(cfg.Goos, cfg.Goarch))
	if b.gccSupportsFlag(compiler, "-fsplit-stack") {
		defs = append(defs, "-fsplit-stack")
	}
	defs = tools.maybePIC(defs)
	if b.gccSupportsFlag(compiler, "-ffile-prefix-map=a=b") {
		defs = append(defs, "-ffile-prefix-map="+base.Cwd()+"=.")
		defs = append(defs, "-ffile-prefix-map="+b.WorkDir+"=/tmp/go-build")
	} else if b.gccSupportsFlag(compiler, "-fdebug-prefix-map=a=b") {
		defs = append(defs, "-fdebug-prefix-map="+b.WorkDir+"=/tmp/go-build")
	}
	if b.gccSupportsFlag(compiler, "-gno-record-gcc-switches") {
		defs = append(defs, "-gno-record-gcc-switches")
	}
	return b.Shell(a).run(p.Dir, p.ImportPath, nil, compiler, "-Wall", "-g",
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

var gccgoToSymbolFuncOnce sync.Once
var gccgoToSymbolFunc func(string) string

func (tools gccgoToolchain) gccgoCleanPkgpath(b *Builder, p *load.Package) string {
	gccgoToSymbolFuncOnce.Do(func() {
		tmpdir := b.WorkDir
		if cfg.BuildN {
			tmpdir = os.TempDir()
		}
		fn, err := pkgpath.ToSymbolFunc(tools.compiler(), tmpdir)
		if err != nil {
			fmt.Fprintf(os.Stderr, "cmd/go: %v\n", err)
			base.SetExitStatus(2)
			base.Exit()
		}
		gccgoToSymbolFunc = fn
	})

	return gccgoToSymbolFunc(gccgoPkgpath(p))
}

var (
	gccgoSupportsCgoIncompleteOnce sync.Once
	gccgoSupportsCgoIncomplete     bool
)

const gccgoSupportsCgoIncompleteCode = `
package p

import "runtime/cgo"

type I cgo.Incomplete
`

// supportsCgoIncomplete reports whether the gccgo/GoLLVM compiler
// being used supports cgo.Incomplete, which was added in GCC 13.
//
// This takes an Action only for output reporting purposes.
// The result value is unrelated to the Action.
func (tools gccgoToolchain) supportsCgoIncomplete(b *Builder, a *Action) bool {
	gccgoSupportsCgoIncompleteOnce.Do(func() {
		sh := b.Shell(a)

		fail := func(err error) {
			fmt.Fprintf(os.Stderr, "cmd/go: %v\n", err)
			base.SetExitStatus(2)
			base.Exit()
		}

		tmpdir := b.WorkDir
		if cfg.BuildN {
			tmpdir = os.TempDir()
		}
		f, err := os.CreateTemp(tmpdir, "*_gccgo_cgoincomplete.go")
		if err != nil {
			fail(err)
		}
		fn := f.Name()
		f.Close()
		defer os.Remove(fn)

		if err := os.WriteFile(fn, []byte(gccgoSupportsCgoIncompleteCode), 0644); err != nil {
			fail(err)
		}

		on := strings.TrimSuffix(fn, ".go") + ".o"
		if cfg.BuildN || cfg.BuildX {
			sh.ShowCmd(tmpdir, "%s -c -o %s %s || true", tools.compiler(), on, fn)
			// Since this function affects later builds,
			// and only generates temporary files,
			// we run the command even with -n.
		}
		cmd := exec.Command(tools.compiler(), "-c", "-o", on, fn)
		cmd.Dir = tmpdir
		var buf bytes.Buffer
		cmd.Stdout = &buf
		cmd.Stderr = &buf
		err = cmd.Run()
		gccgoSupportsCgoIncomplete = err == nil
		if cfg.BuildN || cfg.BuildX {
			// Show output. We always pass a nil err because errors are an
			// expected outcome in this case.
			desc := sh.fmtCmd(tmpdir, "%s -c -o %s %s", tools.compiler(), on, fn)
			sh.reportCmd(desc, tmpdir, buf.Bytes(), nil)
		}
	})
	return gccgoSupportsCgoIncomplete
}
