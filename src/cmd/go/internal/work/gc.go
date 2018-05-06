// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package work

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/load"
	"cmd/go/internal/str"
	"cmd/internal/objabi"
	"crypto/sha1"
)

// The Go toolchain.

type gcToolchain struct{}

func (gcToolchain) compiler() string {
	return base.Tool("compile")
}

func (gcToolchain) linker() string {
	return base.Tool("link")
}

func (gcToolchain) gc(b *Builder, a *Action, archive string, importcfg []byte, asmhdr bool, gofiles []string) (ofile string, output []byte, err error) {
	p := a.Package
	objdir := a.Objdir
	if archive != "" {
		ofile = archive
	} else {
		out := "_go_.o"
		ofile = objdir + out
	}

	pkgpath := p.ImportPath
	if cfg.BuildBuildmode == "plugin" {
		pkgpath = pluginPath(a)
	} else if p.Name == "main" && !p.Internal.ForceLibrary {
		pkgpath = "main"
	}
	gcargs := []string{"-p", pkgpath}
	if p.Standard {
		gcargs = append(gcargs, "-std")
	}
	compilingRuntime := p.Standard && (p.ImportPath == "runtime" || strings.HasPrefix(p.ImportPath, "runtime/internal"))
	// The runtime package imports a couple of general internal packages.
	if p.Standard && (p.ImportPath == "internal/cpu" || p.ImportPath == "internal/bytealg") {
		compilingRuntime = true
	}
	if compilingRuntime {
		// runtime compiles with a special gc flag to check for
		// memory allocations that are invalid in the runtime package,
		// and to implement some special compiler pragmas.
		gcargs = append(gcargs, "-+")
	}

	// If we're giving the compiler the entire package (no C etc files), tell it that,
	// so that it can give good error messages about forward declarations.
	// Exceptions: a few standard packages have forward declarations for
	// pieces supplied behind-the-scenes by package runtime.
	extFiles := len(p.CgoFiles) + len(p.CFiles) + len(p.CXXFiles) + len(p.MFiles) + len(p.FFiles) + len(p.SFiles) + len(p.SysoFiles) + len(p.SwigFiles) + len(p.SwigCXXFiles)
	if p.Standard {
		switch p.ImportPath {
		case "bytes", "internal/poll", "net", "os", "runtime/pprof", "runtime/trace", "sync", "syscall", "time":
			extFiles++
		}
	}
	if extFiles == 0 {
		gcargs = append(gcargs, "-complete")
	}
	if cfg.BuildContext.InstallSuffix != "" {
		gcargs = append(gcargs, "-installsuffix", cfg.BuildContext.InstallSuffix)
	}
	if a.buildID != "" {
		gcargs = append(gcargs, "-buildid", a.buildID)
	}
	platform := cfg.Goos + "/" + cfg.Goarch
	if p.Internal.OmitDebug || platform == "nacl/amd64p32" || cfg.Goos == "plan9" || cfg.Goarch == "wasm" {
		gcargs = append(gcargs, "-dwarf=false")
	}
	if strings.HasPrefix(runtimeVersion, "go1") && !strings.Contains(os.Args[0], "go_bootstrap") {
		gcargs = append(gcargs, "-goversion", runtimeVersion)
	}

	gcflags := str.StringList(forcedGcflags, p.Internal.Gcflags)
	if compilingRuntime {
		// Remove -N, if present.
		// It is not possible to build the runtime with no optimizations,
		// because the compiler cannot eliminate enough write barriers.
		for i := 0; i < len(gcflags); i++ {
			if gcflags[i] == "-N" {
				copy(gcflags[i:], gcflags[i+1:])
				gcflags = gcflags[:len(gcflags)-1]
				i--
			}
		}
	}

	args := []interface{}{cfg.BuildToolexec, base.Tool("compile"), "-o", ofile, "-trimpath", trimDir(a.Objdir), gcflags, gcargs, "-D", p.Internal.LocalPrefix}
	if importcfg != nil {
		if err := b.writeFile(objdir+"importcfg", importcfg); err != nil {
			return "", nil, err
		}
		args = append(args, "-importcfg", objdir+"importcfg")
	}
	if ofile == archive {
		args = append(args, "-pack")
	}
	if asmhdr {
		args = append(args, "-asmhdr", objdir+"go_asm.h")
	}

	// Add -c=N to use concurrent backend compilation, if possible.
	if c := gcBackendConcurrency(gcflags); c > 1 {
		args = append(args, fmt.Sprintf("-c=%d", c))
	}

	for _, f := range gofiles {
		args = append(args, mkAbs(p.Dir, f))
	}

	output, err = b.runOut(p.Dir, nil, args...)
	return ofile, output, err
}

// gcBackendConcurrency returns the backend compiler concurrency level for a package compilation.
func gcBackendConcurrency(gcflags []string) int {
	// First, check whether we can use -c at all for this compilation.
	canDashC := concurrentGCBackendCompilationEnabledByDefault

	switch e := os.Getenv("GO19CONCURRENTCOMPILATION"); e {
	case "0":
		canDashC = false
	case "1":
		canDashC = true
	case "":
		// Not set. Use default.
	default:
		log.Fatalf("GO19CONCURRENTCOMPILATION must be 0, 1, or unset, got %q", e)
	}

CheckFlags:
	for _, flag := range gcflags {
		// Concurrent compilation is presumed incompatible with any gcflags,
		// except for a small whitelist of commonly used flags.
		// If the user knows better, they can manually add their own -c to the gcflags.
		switch flag {
		case "-N", "-l", "-S", "-B", "-C", "-I":
			// OK
		default:
			canDashC = false
			break CheckFlags
		}
	}

	// TODO: Test and delete these conditions.
	if objabi.Fieldtrack_enabled != 0 || objabi.Preemptibleloops_enabled != 0 || objabi.Clobberdead_enabled != 0 {
		canDashC = false
	}

	if !canDashC {
		return 1
	}

	// Decide how many concurrent backend compilations to allow.
	//
	// If we allow too many, in theory we might end up with p concurrent processes,
	// each with c concurrent backend compiles, all fighting over the same resources.
	// However, in practice, that seems not to happen too much.
	// Most build graphs are surprisingly serial, so p==1 for much of the build.
	// Furthermore, concurrent backend compilation is only enabled for a part
	// of the overall compiler execution, so c==1 for much of the build.
	// So don't worry too much about that interaction for now.
	//
	// However, in practice, setting c above 4 tends not to help very much.
	// See the analysis in CL 41192.
	//
	// TODO(josharian): attempt to detect whether this particular compilation
	// is likely to be a bottleneck, e.g. when:
	//   - it has no successor packages to compile (usually package main)
	//   - all paths through the build graph pass through it
	//   - critical path scheduling says it is high priority
	// and in such a case, set c to runtime.NumCPU.
	// We do this now when p==1.
	if cfg.BuildP == 1 {
		// No process parallelism. Max out c.
		return runtime.NumCPU()
	}
	// Some process parallelism. Set c to min(4, numcpu).
	c := 4
	if ncpu := runtime.NumCPU(); ncpu < c {
		c = ncpu
	}
	return c
}

func trimDir(dir string) string {
	if len(dir) > 1 && dir[len(dir)-1] == filepath.Separator {
		dir = dir[:len(dir)-1]
	}
	return dir
}

func (gcToolchain) asm(b *Builder, a *Action, sfiles []string) ([]string, error) {
	p := a.Package
	// Add -I pkg/GOOS_GOARCH so #include "textflag.h" works in .s files.
	inc := filepath.Join(cfg.GOROOT, "pkg", "include")
	args := []interface{}{cfg.BuildToolexec, base.Tool("asm"), "-trimpath", trimDir(a.Objdir), "-I", a.Objdir, "-I", inc, "-D", "GOOS_" + cfg.Goos, "-D", "GOARCH_" + cfg.Goarch, forcedAsmflags, p.Internal.Asmflags}
	if p.ImportPath == "runtime" && cfg.Goarch == "386" {
		for _, arg := range forcedAsmflags {
			if arg == "-dynlink" {
				args = append(args, "-D=GOBUILDMODE_shared=1")
			}
		}
	}

	if cfg.Goarch == "mips" || cfg.Goarch == "mipsle" {
		// Define GOMIPS_value from cfg.GOMIPS.
		args = append(args, "-D", "GOMIPS_"+cfg.GOMIPS)
	}

	if cfg.Goarch == "mips64" || cfg.Goarch == "mips64le" {
		// Define GOMIPS64_value from cfg.GOMIPS64.
		args = append(args, "-D", "GOMIPS64_"+cfg.GOMIPS64)
	}

	var ofiles []string
	for _, sfile := range sfiles {
		ofile := a.Objdir + sfile[:len(sfile)-len(".s")] + ".o"
		ofiles = append(ofiles, ofile)
		args1 := append(args, "-o", ofile, mkAbs(p.Dir, sfile))
		if err := b.run(a, p.Dir, p.ImportPath, nil, args1...); err != nil {
			return nil, err
		}
	}
	return ofiles, nil
}

// toolVerify checks that the command line args writes the same output file
// if run using newTool instead.
// Unused now but kept around for future use.
func toolVerify(a *Action, b *Builder, p *load.Package, newTool string, ofile string, args []interface{}) error {
	newArgs := make([]interface{}, len(args))
	copy(newArgs, args)
	newArgs[1] = base.Tool(newTool)
	newArgs[3] = ofile + ".new" // x.6 becomes x.6.new
	if err := b.run(a, p.Dir, p.ImportPath, nil, newArgs...); err != nil {
		return err
	}
	data1, err := ioutil.ReadFile(ofile)
	if err != nil {
		return err
	}
	data2, err := ioutil.ReadFile(ofile + ".new")
	if err != nil {
		return err
	}
	if !bytes.Equal(data1, data2) {
		return fmt.Errorf("%s and %s produced different output files:\n%s\n%s", filepath.Base(args[1].(string)), newTool, strings.Join(str.StringList(args...), " "), strings.Join(str.StringList(newArgs...), " "))
	}
	os.Remove(ofile + ".new")
	return nil
}

func (gcToolchain) pack(b *Builder, a *Action, afile string, ofiles []string) error {
	var absOfiles []string
	for _, f := range ofiles {
		absOfiles = append(absOfiles, mkAbs(a.Objdir, f))
	}
	absAfile := mkAbs(a.Objdir, afile)

	// The archive file should have been created by the compiler.
	// Since it used to not work that way, verify.
	if !cfg.BuildN {
		if _, err := os.Stat(absAfile); err != nil {
			base.Fatalf("os.Stat of archive file failed: %v", err)
		}
	}

	p := a.Package
	if cfg.BuildN || cfg.BuildX {
		cmdline := str.StringList(base.Tool("pack"), "r", absAfile, absOfiles)
		b.Showcmd(p.Dir, "%s # internal", joinUnambiguously(cmdline))
	}
	if cfg.BuildN {
		return nil
	}
	if err := packInternal(absAfile, absOfiles); err != nil {
		b.showOutput(a, p.Dir, p.Desc(), err.Error()+"\n")
		return errPrintedOutput
	}
	return nil
}

func packInternal(afile string, ofiles []string) error {
	dst, err := os.OpenFile(afile, os.O_WRONLY|os.O_APPEND, 0)
	if err != nil {
		return err
	}
	defer dst.Close() // only for error returns or panics
	w := bufio.NewWriter(dst)

	for _, ofile := range ofiles {
		src, err := os.Open(ofile)
		if err != nil {
			return err
		}
		fi, err := src.Stat()
		if err != nil {
			src.Close()
			return err
		}
		// Note: Not using %-16.16s format because we care
		// about bytes, not runes.
		name := fi.Name()
		if len(name) > 16 {
			name = name[:16]
		} else {
			name += strings.Repeat(" ", 16-len(name))
		}
		size := fi.Size()
		fmt.Fprintf(w, "%s%-12d%-6d%-6d%-8o%-10d`\n",
			name, 0, 0, 0, 0644, size)
		n, err := io.Copy(w, src)
		src.Close()
		if err == nil && n < size {
			err = io.ErrUnexpectedEOF
		} else if err == nil && n > size {
			err = fmt.Errorf("file larger than size reported by stat")
		}
		if err != nil {
			return fmt.Errorf("copying %s to %s: %v", ofile, afile, err)
		}
		if size&1 != 0 {
			w.WriteByte(0)
		}
	}

	if err := w.Flush(); err != nil {
		return err
	}
	return dst.Close()
}

// setextld sets the appropriate linker flags for the specified compiler.
func setextld(ldflags []string, compiler []string) []string {
	for _, f := range ldflags {
		if f == "-extld" || strings.HasPrefix(f, "-extld=") {
			// don't override -extld if supplied
			return ldflags
		}
	}
	ldflags = append(ldflags, "-extld="+compiler[0])
	if len(compiler) > 1 {
		extldflags := false
		add := strings.Join(compiler[1:], " ")
		for i, f := range ldflags {
			if f == "-extldflags" && i+1 < len(ldflags) {
				ldflags[i+1] = add + " " + ldflags[i+1]
				extldflags = true
				break
			} else if strings.HasPrefix(f, "-extldflags=") {
				ldflags[i] = "-extldflags=" + add + " " + ldflags[i][len("-extldflags="):]
				extldflags = true
				break
			}
		}
		if !extldflags {
			ldflags = append(ldflags, "-extldflags="+add)
		}
	}
	return ldflags
}

// pluginPath computes the package path for a plugin main package.
//
// This is typically the import path of the main package p, unless the
// plugin is being built directly from source files. In that case we
// combine the package build ID with the contents of the main package
// source files. This allows us to identify two different plugins
// built from two source files with the same name.
func pluginPath(a *Action) string {
	p := a.Package
	if p.ImportPath != "command-line-arguments" {
		return p.ImportPath
	}
	h := sha1.New()
	fmt.Fprintf(h, "build ID: %s\n", a.buildID)
	for _, file := range str.StringList(p.GoFiles, p.CgoFiles, p.SFiles) {
		data, err := ioutil.ReadFile(filepath.Join(p.Dir, file))
		if err != nil {
			base.Fatalf("go: %s", err)
		}
		h.Write(data)
	}
	return fmt.Sprintf("plugin/unnamed-%x", h.Sum(nil))
}

func (gcToolchain) ld(b *Builder, root *Action, out, importcfg, mainpkg string) error {
	cxx := len(root.Package.CXXFiles) > 0 || len(root.Package.SwigCXXFiles) > 0
	for _, a := range root.Deps {
		if a.Package != nil && (len(a.Package.CXXFiles) > 0 || len(a.Package.SwigCXXFiles) > 0) {
			cxx = true
		}
	}
	var ldflags []string
	if cfg.BuildContext.InstallSuffix != "" {
		ldflags = append(ldflags, "-installsuffix", cfg.BuildContext.InstallSuffix)
	}
	if root.Package.Internal.OmitDebug {
		ldflags = append(ldflags, "-s", "-w")
	}
	if cfg.BuildBuildmode == "plugin" {
		ldflags = append(ldflags, "-pluginpath", pluginPath(root))
	}

	// Store BuildID inside toolchain binaries as a unique identifier of the
	// tool being run, for use by content-based staleness determination.
	if root.Package.Goroot && strings.HasPrefix(root.Package.ImportPath, "cmd/") {
		ldflags = append(ldflags, "-X=cmd/internal/objabi.buildID="+root.buildID)
	}

	// If the user has not specified the -extld option, then specify the
	// appropriate linker. In case of C++ code, use the compiler named
	// by the CXX environment variable or defaultCXX if CXX is not set.
	// Else, use the CC environment variable and defaultCC as fallback.
	var compiler []string
	if cxx {
		compiler = envList("CXX", cfg.DefaultCXX(cfg.Goos, cfg.Goarch))
	} else {
		compiler = envList("CC", cfg.DefaultCC(cfg.Goos, cfg.Goarch))
	}
	ldflags = append(ldflags, "-buildmode="+ldBuildmode)
	if root.buildID != "" {
		ldflags = append(ldflags, "-buildid="+root.buildID)
	}
	ldflags = append(ldflags, forcedLdflags...)
	ldflags = append(ldflags, root.Package.Internal.Ldflags...)
	ldflags = setextld(ldflags, compiler)

	// On OS X when using external linking to build a shared library,
	// the argument passed here to -o ends up recorded in the final
	// shared library in the LC_ID_DYLIB load command.
	// To avoid putting the temporary output directory name there
	// (and making the resulting shared library useless),
	// run the link in the output directory so that -o can name
	// just the final path element.
	// On Windows, DLL file name is recorded in PE file
	// export section, so do like on OS X.
	dir := "."
	if (cfg.Goos == "darwin" || cfg.Goos == "windows") && cfg.BuildBuildmode == "c-shared" {
		dir, out = filepath.Split(out)
	}

	return b.run(root, dir, root.Package.ImportPath, nil, cfg.BuildToolexec, base.Tool("link"), "-o", out, "-importcfg", importcfg, ldflags, mainpkg)
}

func (gcToolchain) ldShared(b *Builder, root *Action, toplevelactions []*Action, out, importcfg string, allactions []*Action) error {
	ldflags := []string{"-installsuffix", cfg.BuildContext.InstallSuffix}
	ldflags = append(ldflags, "-buildmode=shared")
	ldflags = append(ldflags, forcedLdflags...)
	ldflags = append(ldflags, root.Package.Internal.Ldflags...)
	cxx := false
	for _, a := range allactions {
		if a.Package != nil && (len(a.Package.CXXFiles) > 0 || len(a.Package.SwigCXXFiles) > 0) {
			cxx = true
		}
	}
	// If the user has not specified the -extld option, then specify the
	// appropriate linker. In case of C++ code, use the compiler named
	// by the CXX environment variable or defaultCXX if CXX is not set.
	// Else, use the CC environment variable and defaultCC as fallback.
	var compiler []string
	if cxx {
		compiler = envList("CXX", cfg.DefaultCXX(cfg.Goos, cfg.Goarch))
	} else {
		compiler = envList("CC", cfg.DefaultCC(cfg.Goos, cfg.Goarch))
	}
	ldflags = setextld(ldflags, compiler)
	for _, d := range toplevelactions {
		if !strings.HasSuffix(d.Target, ".a") { // omit unsafe etc and actions for other shared libraries
			continue
		}
		ldflags = append(ldflags, d.Package.ImportPath+"="+d.Target)
	}
	return b.run(root, ".", out, nil, cfg.BuildToolexec, base.Tool("link"), "-o", out, "-importcfg", importcfg, ldflags)
}

func (gcToolchain) cc(b *Builder, a *Action, ofile, cfile string) error {
	return fmt.Errorf("%s: C source files not supported without cgo", mkAbs(a.Package.Dir, cfile))
}
