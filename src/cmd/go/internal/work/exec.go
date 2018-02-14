// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Action graph execution.

package work

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"cmd/go/internal/base"
	"cmd/go/internal/cache"
	"cmd/go/internal/cfg"
	"cmd/go/internal/load"
	"cmd/go/internal/str"
)

// actionList returns the list of actions in the dag rooted at root
// as visited in a depth-first post-order traversal.
func actionList(root *Action) []*Action {
	seen := map[*Action]bool{}
	all := []*Action{}
	var walk func(*Action)
	walk = func(a *Action) {
		if seen[a] {
			return
		}
		seen[a] = true
		for _, a1 := range a.Deps {
			walk(a1)
		}
		all = append(all, a)
	}
	walk(root)
	return all
}

// do runs the action graph rooted at root.
func (b *Builder) Do(root *Action) {
	if c := cache.Default(); c != nil && !b.ComputeStaleOnly {
		// If we're doing real work, take time at the end to trim the cache.
		defer c.Trim()
	}

	// Build list of all actions, assigning depth-first post-order priority.
	// The original implementation here was a true queue
	// (using a channel) but it had the effect of getting
	// distracted by low-level leaf actions to the detriment
	// of completing higher-level actions. The order of
	// work does not matter much to overall execution time,
	// but when running "go test std" it is nice to see each test
	// results as soon as possible. The priorities assigned
	// ensure that, all else being equal, the execution prefers
	// to do what it would have done first in a simple depth-first
	// dependency order traversal.
	all := actionList(root)
	for i, a := range all {
		a.priority = i
	}

	if cfg.DebugActiongraph != "" {
		js := actionGraphJSON(root)
		if err := ioutil.WriteFile(cfg.DebugActiongraph, []byte(js), 0666); err != nil {
			fmt.Fprintf(os.Stderr, "go: writing action graph: %v\n", err)
			base.SetExitStatus(1)
		}
	}

	b.readySema = make(chan bool, len(all))

	// Initialize per-action execution state.
	for _, a := range all {
		for _, a1 := range a.Deps {
			a1.triggers = append(a1.triggers, a)
		}
		a.pending = len(a.Deps)
		if a.pending == 0 {
			b.ready.push(a)
			b.readySema <- true
		}
	}

	// Handle runs a single action and takes care of triggering
	// any actions that are runnable as a result.
	handle := func(a *Action) {
		var err error

		if a.Func != nil && (!a.Failed || a.IgnoreFail) {
			if err == nil {
				err = a.Func(b, a)
			}
		}

		// The actions run in parallel but all the updates to the
		// shared work state are serialized through b.exec.
		b.exec.Lock()
		defer b.exec.Unlock()

		if err != nil {
			if err == errPrintedOutput {
				base.SetExitStatus(2)
			} else {
				base.Errorf("%s", err)
			}
			a.Failed = true
		}

		for _, a0 := range a.triggers {
			if a.Failed {
				a0.Failed = true
			}
			if a0.pending--; a0.pending == 0 {
				b.ready.push(a0)
				b.readySema <- true
			}
		}

		if a == root {
			close(b.readySema)
		}
	}

	var wg sync.WaitGroup

	// Kick off goroutines according to parallelism.
	// If we are using the -n flag (just printing commands)
	// drop the parallelism to 1, both to make the output
	// deterministic and because there is no real work anyway.
	par := cfg.BuildP
	if cfg.BuildN {
		par = 1
	}
	for i := 0; i < par; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case _, ok := <-b.readySema:
					if !ok {
						return
					}
					// Receiving a value from b.readySema entitles
					// us to take from the ready queue.
					b.exec.Lock()
					a := b.ready.pop()
					b.exec.Unlock()
					handle(a)
				case <-base.Interrupted:
					base.SetExitStatus(1)
					return
				}
			}
		}()
	}

	wg.Wait()
}

// buildActionID computes the action ID for a build action.
func (b *Builder) buildActionID(a *Action) cache.ActionID {
	p := a.Package
	h := cache.NewHash("build " + p.ImportPath)

	// Configuration independent of compiler toolchain.
	// Note: buildmode has already been accounted for in buildGcflags
	// and should not be inserted explicitly. Most buildmodes use the
	// same compiler settings and can reuse each other's results.
	// If not, the reason is already recorded in buildGcflags.
	fmt.Fprintf(h, "compile\n")
	// The compiler hides the exact value of $GOROOT
	// when building things in GOROOT,
	// but it does not hide the exact value of $GOPATH.
	// Include the full dir in that case.
	// Assume b.WorkDir is being trimmed properly.
	if !p.Goroot && !strings.HasPrefix(p.Dir, b.WorkDir) {
		fmt.Fprintf(h, "dir %s\n", p.Dir)
	}
	fmt.Fprintf(h, "goos %s goarch %s\n", cfg.Goos, cfg.Goarch)
	fmt.Fprintf(h, "import %q\n", p.ImportPath)
	fmt.Fprintf(h, "omitdebug %v standard %v local %v prefix %q\n", p.Internal.OmitDebug, p.Standard, p.Internal.Local, p.Internal.LocalPrefix)
	if len(p.CgoFiles)+len(p.SwigFiles) > 0 {
		fmt.Fprintf(h, "cgo %q\n", b.toolID("cgo"))
		cppflags, cflags, cxxflags, fflags, _, _ := b.CFlags(p)
		fmt.Fprintf(h, "CC=%q %q %q\n", b.ccExe(), cppflags, cflags)
		if len(p.CXXFiles)+len(p.SwigFiles) > 0 {
			fmt.Fprintf(h, "CXX=%q %q\n", b.cxxExe(), cxxflags)
		}
		if len(p.FFiles) > 0 {
			fmt.Fprintf(h, "FC=%q %q\n", b.fcExe(), fflags)
		}
		// TODO(rsc): Should we include the SWIG version or Fortran/GCC/G++/Objective-C compiler versions?
	}
	if p.Internal.CoverMode != "" {
		fmt.Fprintf(h, "cover %q %q\n", p.Internal.CoverMode, b.toolID("cover"))
	}

	// Configuration specific to compiler toolchain.
	switch cfg.BuildToolchainName {
	default:
		base.Fatalf("buildActionID: unknown build toolchain %q", cfg.BuildToolchainName)
	case "gc":
		fmt.Fprintf(h, "compile %s %q %q\n", b.toolID("compile"), forcedGcflags, p.Internal.Gcflags)
		if len(p.SFiles) > 0 {
			fmt.Fprintf(h, "asm %q %q %q\n", b.toolID("asm"), forcedAsmflags, p.Internal.Asmflags)
		}
		fmt.Fprintf(h, "GO$GOARCH=%s\n", os.Getenv("GO"+strings.ToUpper(cfg.BuildContext.GOARCH))) // GO386, GOARM, etc

		// TODO(rsc): Convince compiler team not to add more magic environment variables,
		// or perhaps restrict the environment variables passed to subprocesses.
		magic := []string{
			"GOCLOBBERDEADHASH",
			"GOSSAFUNC",
			"GO_SSA_PHI_LOC_CUTOFF",
			"GOSSAHASH",
		}
		for _, env := range magic {
			if x := os.Getenv(env); x != "" {
				fmt.Fprintf(h, "magic %s=%s\n", env, x)
			}
		}
		if os.Getenv("GOSSAHASH") != "" {
			for i := 0; ; i++ {
				env := fmt.Sprintf("GOSSAHASH%d", i)
				x := os.Getenv(env)
				if x == "" {
					break
				}
				fmt.Fprintf(h, "magic %s=%s\n", env, x)
			}
		}
		if os.Getenv("GSHS_LOGFILE") != "" {
			// Clumsy hack. Compiler writes to this log file,
			// so do not allow use of cache at all.
			// We will still write to the cache but it will be
			// essentially unfindable.
			fmt.Fprintf(h, "nocache %d\n", time.Now().UnixNano())
		}

	case "gccgo":
		id, err := b.gccgoToolID(BuildToolchain.compiler(), "go")
		if err != nil {
			base.Fatalf("%v", err)
		}
		fmt.Fprintf(h, "compile %s %q %q\n", id, forcedGccgoflags, p.Internal.Gccgoflags)
		fmt.Fprintf(h, "pkgpath %s\n", gccgoPkgpath(p))
		if len(p.SFiles) > 0 {
			id, err = b.gccgoToolID(BuildToolchain.compiler(), "assembler-with-cpp")
			// Ignore error; different assembler versions
			// are unlikely to make any difference anyhow.
			fmt.Fprintf(h, "asm %q\n", id)
		}
	}

	// Input files.
	inputFiles := str.StringList(
		p.GoFiles,
		p.CgoFiles,
		p.CFiles,
		p.CXXFiles,
		p.FFiles,
		p.MFiles,
		p.HFiles,
		p.SFiles,
		p.SysoFiles,
		p.SwigFiles,
		p.SwigCXXFiles,
	)
	for _, file := range inputFiles {
		fmt.Fprintf(h, "file %s %s\n", file, b.fileHash(filepath.Join(p.Dir, file)))
	}
	for _, a1 := range a.Deps {
		p1 := a1.Package
		if p1 != nil {
			fmt.Fprintf(h, "import %s %s\n", p1.ImportPath, contentID(a1.buildID))
		}
	}

	return h.Sum()
}

// build is the action for building a single package.
// Note that any new influence on this logic must be reported in b.buildActionID above as well.
func (b *Builder) build(a *Action) (err error) {
	p := a.Package
	cached := false
	if !p.BinaryOnly {
		if b.useCache(a, p, b.buildActionID(a), p.Target) {
			// If this build triggers a header install, run cgo to get the header.
			// TODO(rsc): Once we can cache multiple file outputs from an action,
			// the header should be cached, and then this awful test can be deleted.
			// Need to look for install header actions depending on this action,
			// or depending on a link that depends on this action.
			needHeader := false
			if (a.Package.UsesCgo() || a.Package.UsesSwig()) && (cfg.BuildBuildmode == "c-archive" || cfg.BuildBuildmode == "c-shared") {
				for _, t1 := range a.triggers {
					if t1.Mode == "install header" {
						needHeader = true
						goto CheckedHeader
					}
				}
				for _, t1 := range a.triggers {
					for _, t2 := range t1.triggers {
						if t2.Mode == "install header" {
							needHeader = true
							goto CheckedHeader
						}
					}
				}
			}
		CheckedHeader:
			if b.ComputeStaleOnly || !a.needVet && !needHeader {
				return nil
			}
			cached = true
		}
		defer b.flushOutput(a)
	}

	defer func() {
		if err != nil && err != errPrintedOutput {
			err = fmt.Errorf("go build %s: %v", a.Package.ImportPath, err)
		}
	}()
	if cfg.BuildN {
		// In -n mode, print a banner between packages.
		// The banner is five lines so that when changes to
		// different sections of the bootstrap script have to
		// be merged, the banners give patch something
		// to use to find its context.
		b.Print("\n#\n# " + a.Package.ImportPath + "\n#\n\n")
	}

	if cfg.BuildV {
		b.Print(a.Package.ImportPath + "\n")
	}

	if a.Package.BinaryOnly {
		_, err := os.Stat(a.Package.Target)
		if err == nil {
			a.built = a.Package.Target
			a.Target = a.Package.Target
			a.buildID = b.fileHash(a.Package.Target)
			a.Package.Stale = false
			a.Package.StaleReason = "binary-only package"
			return nil
		}
		if b.ComputeStaleOnly {
			a.Package.Stale = true
			a.Package.StaleReason = "missing or invalid binary-only package"
			return nil
		}
		return fmt.Errorf("missing or invalid binary-only package")
	}

	if err := b.Mkdir(a.Objdir); err != nil {
		return err
	}
	objdir := a.Objdir

	// make target directory
	dir, _ := filepath.Split(a.Target)
	if dir != "" {
		if err := b.Mkdir(dir); err != nil {
			return err
		}
	}

	var gofiles, cgofiles, cfiles, sfiles, cxxfiles, objects, cgoObjects, pcCFLAGS, pcLDFLAGS []string

	gofiles = append(gofiles, a.Package.GoFiles...)
	cgofiles = append(cgofiles, a.Package.CgoFiles...)
	cfiles = append(cfiles, a.Package.CFiles...)
	sfiles = append(sfiles, a.Package.SFiles...)
	cxxfiles = append(cxxfiles, a.Package.CXXFiles...)

	if a.Package.UsesCgo() || a.Package.UsesSwig() {
		if pcCFLAGS, pcLDFLAGS, err = b.getPkgConfigFlags(a.Package); err != nil {
			return
		}
	}

	// Run SWIG on each .swig and .swigcxx file.
	// Each run will generate two files, a .go file and a .c or .cxx file.
	// The .go file will use import "C" and is to be processed by cgo.
	if a.Package.UsesSwig() {
		outGo, outC, outCXX, err := b.swig(a, a.Package, objdir, pcCFLAGS)
		if err != nil {
			return err
		}
		cgofiles = append(cgofiles, outGo...)
		cfiles = append(cfiles, outC...)
		cxxfiles = append(cxxfiles, outCXX...)
	}

	// If we're doing coverage, preprocess the .go files and put them in the work directory
	if a.Package.Internal.CoverMode != "" {
		for i, file := range str.StringList(gofiles, cgofiles) {
			var sourceFile string
			var coverFile string
			var key string
			if strings.HasSuffix(file, ".cgo1.go") {
				// cgo files have absolute paths
				base := filepath.Base(file)
				sourceFile = file
				coverFile = objdir + base
				key = strings.TrimSuffix(base, ".cgo1.go") + ".go"
			} else {
				sourceFile = filepath.Join(a.Package.Dir, file)
				coverFile = objdir + file
				key = file
			}
			coverFile = strings.TrimSuffix(coverFile, ".go") + ".cover.go"
			cover := a.Package.Internal.CoverVars[key]
			if cover == nil || base.IsTestFile(file) {
				// Not covering this file.
				continue
			}
			if err := b.cover(a, coverFile, sourceFile, 0666, cover.Var); err != nil {
				return err
			}
			if i < len(gofiles) {
				gofiles[i] = coverFile
			} else {
				cgofiles[i-len(gofiles)] = coverFile
			}
		}
	}

	// Run cgo.
	if a.Package.UsesCgo() || a.Package.UsesSwig() {
		// In a package using cgo, cgo compiles the C, C++ and assembly files with gcc.
		// There is one exception: runtime/cgo's job is to bridge the
		// cgo and non-cgo worlds, so it necessarily has files in both.
		// In that case gcc only gets the gcc_* files.
		var gccfiles []string
		gccfiles = append(gccfiles, cfiles...)
		cfiles = nil
		if a.Package.Standard && a.Package.ImportPath == "runtime/cgo" {
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
			sfiles, gccfiles = filter(sfiles, sfiles[:0], gccfiles)
		} else {
			for _, sfile := range sfiles {
				data, err := ioutil.ReadFile(filepath.Join(a.Package.Dir, sfile))
				if err == nil {
					if bytes.HasPrefix(data, []byte("TEXT")) || bytes.Contains(data, []byte("\nTEXT")) ||
						bytes.HasPrefix(data, []byte("DATA")) || bytes.Contains(data, []byte("\nDATA")) ||
						bytes.HasPrefix(data, []byte("GLOBL")) || bytes.Contains(data, []byte("\nGLOBL")) {
						return fmt.Errorf("package using cgo has Go assembly file %s", sfile)
					}
				}
			}
			gccfiles = append(gccfiles, sfiles...)
			sfiles = nil
		}

		outGo, outObj, err := b.cgo(a, base.Tool("cgo"), objdir, pcCFLAGS, pcLDFLAGS, mkAbsFiles(a.Package.Dir, cgofiles), gccfiles, cxxfiles, a.Package.MFiles, a.Package.FFiles)
		if err != nil {
			return err
		}
		if cfg.BuildToolchainName == "gccgo" {
			cgoObjects = append(cgoObjects, a.Objdir+"_cgo_flags")
		}
		cgoObjects = append(cgoObjects, outObj...)
		gofiles = append(gofiles, outGo...)
	}
	if cached && !a.needVet {
		return nil
	}

	// Sanity check only, since Package.load already checked as well.
	if len(gofiles) == 0 {
		return &load.NoGoError{Package: a.Package}
	}

	// Prepare Go vet config if needed.
	var vcfg *vetConfig
	if a.needVet {
		// Pass list of absolute paths to vet,
		// so that vet's error messages will use absolute paths,
		// so that we can reformat them relative to the directory
		// in which the go command is invoked.
		vcfg = &vetConfig{
			Compiler:    cfg.BuildToolchainName,
			Dir:         a.Package.Dir,
			GoFiles:     mkAbsFiles(a.Package.Dir, gofiles),
			ImportPath:  a.Package.ImportPath,
			ImportMap:   make(map[string]string),
			PackageFile: make(map[string]string),
		}
		a.vetCfg = vcfg
		for i, raw := range a.Package.Internal.RawImports {
			final := a.Package.Imports[i]
			vcfg.ImportMap[raw] = final
		}
	}

	// Prepare Go import config.
	// We start it off with a comment so it can't be empty, so icfg.Bytes() below is never nil.
	// It should never be empty anyway, but there have been bugs in the past that resulted
	// in empty configs, which then unfortunately turn into "no config passed to compiler",
	// and the compiler falls back to looking in pkg itself, which mostly works,
	// except when it doesn't.
	var icfg bytes.Buffer
	fmt.Fprintf(&icfg, "# import config\n")

	for i, raw := range a.Package.Internal.RawImports {
		final := a.Package.Imports[i]
		if final != raw {
			fmt.Fprintf(&icfg, "importmap %s=%s\n", raw, final)
		}
	}

	// Compute the list of mapped imports in the vet config
	// so that we can add any missing mappings below.
	var vcfgMapped map[string]bool
	if vcfg != nil {
		vcfgMapped = make(map[string]bool)
		for _, p := range vcfg.ImportMap {
			vcfgMapped[p] = true
		}
	}

	for _, a1 := range a.Deps {
		p1 := a1.Package
		if p1 == nil || p1.ImportPath == "" || a1.built == "" {
			continue
		}
		fmt.Fprintf(&icfg, "packagefile %s=%s\n", p1.ImportPath, a1.built)
		if vcfg != nil {
			// Add import mapping if needed
			// (for imports like "runtime/cgo" that appear only in generated code).
			if !vcfgMapped[p1.ImportPath] {
				vcfg.ImportMap[p1.ImportPath] = p1.ImportPath
			}
			vcfg.PackageFile[p1.ImportPath] = a1.built
		}
	}

	if cached {
		// The cached package file is OK, so we don't need to run the compile.
		// We've only going through the motions to prepare the vet configuration,
		// which is now complete.
		return nil
	}

	// Compile Go.
	objpkg := objdir + "_pkg_.a"
	ofile, out, err := BuildToolchain.gc(b, a, objpkg, icfg.Bytes(), len(sfiles) > 0, gofiles)
	if len(out) > 0 {
		b.showOutput(a, a.Package.Dir, a.Package.ImportPath, b.processOutput(out))
		if err != nil {
			return errPrintedOutput
		}
	}
	if err != nil {
		return err
	}
	if ofile != objpkg {
		objects = append(objects, ofile)
	}

	// Copy .h files named for goos or goarch or goos_goarch
	// to names using GOOS and GOARCH.
	// For example, defs_linux_amd64.h becomes defs_GOOS_GOARCH.h.
	_goos_goarch := "_" + cfg.Goos + "_" + cfg.Goarch
	_goos := "_" + cfg.Goos
	_goarch := "_" + cfg.Goarch
	for _, file := range a.Package.HFiles {
		name, ext := fileExtSplit(file)
		switch {
		case strings.HasSuffix(name, _goos_goarch):
			targ := file[:len(name)-len(_goos_goarch)] + "_GOOS_GOARCH." + ext
			if err := b.copyFile(a, objdir+targ, filepath.Join(a.Package.Dir, file), 0666, true); err != nil {
				return err
			}
		case strings.HasSuffix(name, _goarch):
			targ := file[:len(name)-len(_goarch)] + "_GOARCH." + ext
			if err := b.copyFile(a, objdir+targ, filepath.Join(a.Package.Dir, file), 0666, true); err != nil {
				return err
			}
		case strings.HasSuffix(name, _goos):
			targ := file[:len(name)-len(_goos)] + "_GOOS." + ext
			if err := b.copyFile(a, objdir+targ, filepath.Join(a.Package.Dir, file), 0666, true); err != nil {
				return err
			}
		}
	}

	for _, file := range cfiles {
		out := file[:len(file)-len(".c")] + ".o"
		if err := BuildToolchain.cc(b, a, objdir+out, file); err != nil {
			return err
		}
		objects = append(objects, out)
	}

	// Assemble .s files.
	if len(sfiles) > 0 {
		ofiles, err := BuildToolchain.asm(b, a, sfiles)
		if err != nil {
			return err
		}
		objects = append(objects, ofiles...)
	}

	// For gccgo on ELF systems, we write the build ID as an assembler file.
	// This lets us set the the SHF_EXCLUDE flag.
	// This is read by readGccgoArchive in cmd/internal/buildid/buildid.go.
	if a.buildID != "" && cfg.BuildToolchainName == "gccgo" {
		switch cfg.Goos {
		case "android", "dragonfly", "freebsd", "linux", "netbsd", "openbsd", "solaris":
			asmfile, err := b.gccgoBuildIDELFFile(a)
			if err != nil {
				return err
			}
			ofiles, err := BuildToolchain.asm(b, a, []string{asmfile})
			if err != nil {
				return err
			}
			objects = append(objects, ofiles...)
		}
	}

	// NOTE(rsc): On Windows, it is critically important that the
	// gcc-compiled objects (cgoObjects) be listed after the ordinary
	// objects in the archive. I do not know why this is.
	// https://golang.org/issue/2601
	objects = append(objects, cgoObjects...)

	// Add system object files.
	for _, syso := range a.Package.SysoFiles {
		objects = append(objects, filepath.Join(a.Package.Dir, syso))
	}

	// Pack into archive in objdir directory.
	// If the Go compiler wrote an archive, we only need to add the
	// object files for non-Go sources to the archive.
	// If the Go compiler wrote an archive and the package is entirely
	// Go sources, there is no pack to execute at all.
	if len(objects) > 0 {
		if err := BuildToolchain.pack(b, a, objpkg, objects); err != nil {
			return err
		}
	}

	if err := b.updateBuildID(a, objpkg, true); err != nil {
		return err
	}

	a.built = objpkg
	return nil
}

type vetConfig struct {
	Compiler    string
	Dir         string
	GoFiles     []string
	ImportMap   map[string]string
	PackageFile map[string]string
	ImportPath  string

	SucceedOnTypecheckFailure bool
}

// VetTool is the path to an alternate vet tool binary.
// The caller is expected to set it (if needed) before executing any vet actions.
var VetTool string

// VetFlags are the flags to pass to vet.
// The caller is expected to set them before executing any vet actions.
var VetFlags []string

func (b *Builder) vet(a *Action) error {
	// a.Deps[0] is the build of the package being vetted.
	// a.Deps[1] is the build of the "fmt" package.

	vcfg := a.Deps[0].vetCfg
	if vcfg == nil {
		// Vet config should only be missing if the build failed.
		if !a.Deps[0].Failed {
			return fmt.Errorf("vet config not found")
		}
		return nil
	}

	if vcfg.ImportMap["fmt"] == "" {
		a1 := a.Deps[1]
		vcfg.ImportMap["fmt"] = "fmt"
		vcfg.PackageFile["fmt"] = a1.built
	}

	// During go test, ignore type-checking failures during vet.
	// We only run vet if the compilation has succeeded,
	// so at least for now assume the bug is in vet.
	// We know of at least #18395.
	// TODO(rsc,gri): Try to remove this for Go 1.11.
	vcfg.SucceedOnTypecheckFailure = cfg.CmdName == "test"

	js, err := json.MarshalIndent(vcfg, "", "\t")
	if err != nil {
		return fmt.Errorf("internal error marshaling vet config: %v", err)
	}
	js = append(js, '\n')
	if err := b.writeFile(a.Objdir+"vet.cfg", js); err != nil {
		return err
	}

	var env []string
	if cfg.BuildToolchainName == "gccgo" {
		env = append(env, "GCCGO="+BuildToolchain.compiler())
	}

	p := a.Package
	tool := VetTool
	if tool == "" {
		tool = base.Tool("vet")
	}
	return b.run(a, p.Dir, p.ImportPath, env, cfg.BuildToolexec, tool, VetFlags, a.Objdir+"vet.cfg")
}

// linkActionID computes the action ID for a link action.
func (b *Builder) linkActionID(a *Action) cache.ActionID {
	p := a.Package
	h := cache.NewHash("link " + p.ImportPath)

	// Toolchain-independent configuration.
	fmt.Fprintf(h, "link\n")
	fmt.Fprintf(h, "buildmode %s goos %s goarch %s\n", cfg.BuildBuildmode, cfg.Goos, cfg.Goarch)
	fmt.Fprintf(h, "import %q\n", p.ImportPath)
	fmt.Fprintf(h, "omitdebug %v standard %v local %v prefix %q\n", p.Internal.OmitDebug, p.Standard, p.Internal.Local, p.Internal.LocalPrefix)

	// Toolchain-dependent configuration, shared with b.linkSharedActionID.
	b.printLinkerConfig(h, p)

	// Input files.
	for _, a1 := range a.Deps {
		p1 := a1.Package
		if p1 != nil {
			if a1.built != "" || a1.buildID != "" {
				buildID := a1.buildID
				if buildID == "" {
					buildID = b.buildID(a1.built)
				}
				fmt.Fprintf(h, "packagefile %s=%s\n", p1.ImportPath, contentID(buildID))
			}
			// Because we put package main's full action ID into the binary's build ID,
			// we must also put the full action ID into the binary's action ID hash.
			if p1.Name == "main" {
				fmt.Fprintf(h, "packagemain %s\n", a1.buildID)
			}
			if p1.Shlib != "" {
				fmt.Fprintf(h, "packageshlib %s=%s\n", p1.ImportPath, contentID(b.buildID(p1.Shlib)))
			}
		}
	}

	return h.Sum()
}

// printLinkerConfig prints the linker config into the hash h,
// as part of the computation of a linker-related action ID.
func (b *Builder) printLinkerConfig(h io.Writer, p *load.Package) {
	switch cfg.BuildToolchainName {
	default:
		base.Fatalf("linkActionID: unknown toolchain %q", cfg.BuildToolchainName)

	case "gc":
		fmt.Fprintf(h, "link %s %q %s\n", b.toolID("link"), forcedLdflags, ldBuildmode)
		if p != nil {
			fmt.Fprintf(h, "linkflags %q\n", p.Internal.Ldflags)
		}
		fmt.Fprintf(h, "GO$GOARCH=%s\n", os.Getenv("GO"+strings.ToUpper(cfg.BuildContext.GOARCH))) // GO386, GOARM, etc

		// The linker writes source file paths that say GOROOT_FINAL.
		fmt.Fprintf(h, "GOROOT=%s\n", cfg.GOROOT_FINAL)

		// TODO(rsc): Convince linker team not to add more magic environment variables,
		// or perhaps restrict the environment variables passed to subprocesses.
		magic := []string{
			"GO_EXTLINK_ENABLED",
		}
		for _, env := range magic {
			if x := os.Getenv(env); x != "" {
				fmt.Fprintf(h, "magic %s=%s\n", env, x)
			}
		}

		// TODO(rsc): Do cgo settings and flags need to be included?
		// Or external linker settings and flags?

	case "gccgo":
		id, err := b.gccgoToolID(BuildToolchain.linker(), "go")
		if err != nil {
			base.Fatalf("%v", err)
		}
		fmt.Fprintf(h, "link %s %s\n", id, ldBuildmode)
		// TODO(iant): Should probably include cgo flags here.
	}
}

// link is the action for linking a single command.
// Note that any new influence on this logic must be reported in b.linkActionID above as well.
func (b *Builder) link(a *Action) (err error) {
	if b.useCache(a, a.Package, b.linkActionID(a), a.Package.Target) {
		return nil
	}
	defer b.flushOutput(a)

	if err := b.Mkdir(a.Objdir); err != nil {
		return err
	}

	importcfg := a.Objdir + "importcfg.link"
	if err := b.writeLinkImportcfg(a, importcfg); err != nil {
		return err
	}

	// make target directory
	dir, _ := filepath.Split(a.Target)
	if dir != "" {
		if err := b.Mkdir(dir); err != nil {
			return err
		}
	}

	if err := BuildToolchain.ld(b, a, a.Target, importcfg, a.Deps[0].built); err != nil {
		return err
	}

	// Update the binary with the final build ID.
	// But if OmitDebug is set, don't rewrite the binary, because we set OmitDebug
	// on binaries that we are going to run and then delete.
	// There's no point in doing work on such a binary.
	// Worse, opening the binary for write here makes it
	// essentially impossible to safely fork+exec due to a fundamental
	// incompatibility between ETXTBSY and threads on modern Unix systems.
	// See golang.org/issue/22220.
	// We still call updateBuildID to update a.buildID, which is important
	// for test result caching, but passing rewrite=false (final arg)
	// means we don't actually rewrite the binary, nor store the
	// result into the cache.
	// Not calling updateBuildID means we also don't insert these
	// binaries into the build object cache. That's probably a net win:
	// less cache space wasted on large binaries we are not likely to
	// need again. (On the other hand it does make repeated go test slower.)
	if err := b.updateBuildID(a, a.Target, !a.Package.Internal.OmitDebug); err != nil {
		return err
	}

	a.built = a.Target
	return nil
}

func (b *Builder) writeLinkImportcfg(a *Action, file string) error {
	// Prepare Go import cfg.
	var icfg bytes.Buffer
	for _, a1 := range a.Deps {
		p1 := a1.Package
		if p1 == nil {
			continue
		}
		fmt.Fprintf(&icfg, "packagefile %s=%s\n", p1.ImportPath, a1.built)
		if p1.Shlib != "" {
			fmt.Fprintf(&icfg, "packageshlib %s=%s\n", p1.ImportPath, p1.Shlib)
		}
	}
	return b.writeFile(file, icfg.Bytes())
}

// PkgconfigCmd returns a pkg-config binary name
// defaultPkgConfig is defined in zdefaultcc.go, written by cmd/dist.
func (b *Builder) PkgconfigCmd() string {
	return envList("PKG_CONFIG", cfg.DefaultPkgConfig)[0]
}

// splitPkgConfigOutput parses the pkg-config output into a slice of
// flags. pkg-config always uses \ to escape special characters.
func splitPkgConfigOutput(out []byte) []string {
	if len(out) == 0 {
		return nil
	}
	var flags []string
	flag := make([]byte, len(out))
	r, w := 0, 0
	for r < len(out) {
		switch out[r] {
		case ' ', '\t', '\r', '\n':
			if w > 0 {
				flags = append(flags, string(flag[:w]))
			}
			w = 0
		case '\\':
			r++
			fallthrough
		default:
			if r < len(out) {
				flag[w] = out[r]
				w++
			}
		}
		r++
	}
	if w > 0 {
		flags = append(flags, string(flag[:w]))
	}
	return flags
}

// Calls pkg-config if needed and returns the cflags/ldflags needed to build the package.
func (b *Builder) getPkgConfigFlags(p *load.Package) (cflags, ldflags []string, err error) {
	if pkgs := p.CgoPkgConfig; len(pkgs) > 0 {
		var pcflags []string
		for len(pkgs) > 0 && strings.HasPrefix(pkgs[0], "--") {
			pcflags = append(pcflags, pkgs[0])
			pkgs = pkgs[1:]
		}
		for _, pkg := range pkgs {
			if !load.SafeArg(pkg) {
				return nil, nil, fmt.Errorf("invalid pkg-config package name: %s", pkg)
			}
		}
		var out []byte
		out, err = b.runOut(p.Dir, p.ImportPath, nil, b.PkgconfigCmd(), "--cflags", pcflags, "--", pkgs)
		if err != nil {
			b.showOutput(nil, p.Dir, b.PkgconfigCmd()+" --cflags "+strings.Join(pcflags, " ")+strings.Join(pkgs, " "), string(out))
			b.Print(err.Error() + "\n")
			return nil, nil, errPrintedOutput
		}
		if len(out) > 0 {
			cflags = splitPkgConfigOutput(out)
			if err := checkCompilerFlags("CFLAGS", "pkg-config --cflags", cflags); err != nil {
				return nil, nil, err
			}
		}
		out, err = b.runOut(p.Dir, p.ImportPath, nil, b.PkgconfigCmd(), "--libs", pcflags, "--", pkgs)
		if err != nil {
			b.showOutput(nil, p.Dir, b.PkgconfigCmd()+" --libs "+strings.Join(pcflags, " ")+strings.Join(pkgs, " "), string(out))
			b.Print(err.Error() + "\n")
			return nil, nil, errPrintedOutput
		}
		if len(out) > 0 {
			ldflags = strings.Fields(string(out))
			if err := checkLinkerFlags("LDFLAGS", "pkg-config --libs", ldflags); err != nil {
				return nil, nil, err
			}
		}
	}

	return
}

func (b *Builder) installShlibname(a *Action) error {
	// TODO: BuildN
	a1 := a.Deps[0]
	err := ioutil.WriteFile(a.Target, []byte(filepath.Base(a1.Target)+"\n"), 0666)
	if err != nil {
		return err
	}
	if cfg.BuildX {
		b.Showcmd("", "echo '%s' > %s # internal", filepath.Base(a1.Target), a.Target)
	}
	return nil
}

func (b *Builder) linkSharedActionID(a *Action) cache.ActionID {
	h := cache.NewHash("linkShared")

	// Toolchain-independent configuration.
	fmt.Fprintf(h, "linkShared\n")
	fmt.Fprintf(h, "goos %s goarch %s\n", cfg.Goos, cfg.Goarch)

	// Toolchain-dependent configuration, shared with b.linkActionID.
	b.printLinkerConfig(h, nil)

	// Input files.
	for _, a1 := range a.Deps {
		p1 := a1.Package
		if a1.built == "" {
			continue
		}
		if p1 != nil {
			fmt.Fprintf(h, "packagefile %s=%s\n", p1.ImportPath, contentID(b.buildID(a1.built)))
			if p1.Shlib != "" {
				fmt.Fprintf(h, "packageshlib %s=%s\n", p1.ImportPath, contentID(b.buildID(p1.Shlib)))
			}
		}
	}
	// Files named on command line are special.
	for _, a1 := range a.Deps[0].Deps {
		p1 := a1.Package
		fmt.Fprintf(h, "top %s=%s\n", p1.ImportPath, contentID(b.buildID(a1.built)))
	}

	return h.Sum()
}

func (b *Builder) linkShared(a *Action) (err error) {
	if b.useCache(a, nil, b.linkSharedActionID(a), a.Target) {
		return nil
	}
	defer b.flushOutput(a)

	if err := b.Mkdir(a.Objdir); err != nil {
		return err
	}

	importcfg := a.Objdir + "importcfg.link"
	if err := b.writeLinkImportcfg(a, importcfg); err != nil {
		return err
	}

	// TODO(rsc): There is a missing updateBuildID here,
	// but we have to decide where to store the build ID in these files.
	a.built = a.Target
	return BuildToolchain.ldShared(b, a, a.Deps[0].Deps, a.Target, importcfg, a.Deps)
}

// BuildInstallFunc is the action for installing a single package or executable.
func BuildInstallFunc(b *Builder, a *Action) (err error) {
	defer func() {
		if err != nil && err != errPrintedOutput {
			// a.Package == nil is possible for the go install -buildmode=shared
			// action that installs libmangledname.so, which corresponds to
			// a list of packages, not just one.
			sep, path := "", ""
			if a.Package != nil {
				sep, path = " ", a.Package.ImportPath
			}
			err = fmt.Errorf("go %s%s%s: %v", cfg.CmdName, sep, path, err)
		}
	}()

	a1 := a.Deps[0]
	a.buildID = a1.buildID

	// If we are using the eventual install target as an up-to-date
	// cached copy of the thing we built, then there's no need to
	// copy it into itself (and that would probably fail anyway).
	// In this case a1.built == a.Target because a1.built == p.Target,
	// so the built target is not in the a1.Objdir tree that b.cleanup(a1) removes.
	if a1.built == a.Target {
		a.built = a.Target
		b.cleanup(a1)
		// Whether we're smart enough to avoid a complete rebuild
		// depends on exactly what the staleness and rebuild algorithms
		// are, as well as potentially the state of the Go build cache.
		// We don't really want users to be able to infer (or worse start depending on)
		// those details from whether the modification time changes during
		// "go install", so do a best-effort update of the file times to make it
		// look like we rewrote a.Target even if we did not. Updating the mtime
		// may also help other mtime-based systems that depend on our
		// previous mtime updates that happened more often.
		// This is still not perfect - we ignore the error result, and if the file was
		// unwritable for some reason then pretending to have written it is also
		// confusing - but it's probably better than not doing the mtime update.
		//
		// But don't do that for the special case where building an executable
		// with -linkshared implicitly installs all its dependent libraries.
		// We want to hide that awful detail as much as possible, so don't
		// advertise it by touching the mtimes (usually the libraries are up
		// to date).
		if !a.buggyInstall {
			now := time.Now()
			os.Chtimes(a.Target, now, now)
		}
		return nil
	}
	if b.ComputeStaleOnly {
		return nil
	}

	if err := b.Mkdir(a.Objdir); err != nil {
		return err
	}

	perm := os.FileMode(0666)
	if a1.Mode == "link" {
		switch cfg.BuildBuildmode {
		case "c-archive", "c-shared", "plugin":
		default:
			perm = 0777
		}
	}

	// make target directory
	dir, _ := filepath.Split(a.Target)
	if dir != "" {
		if err := b.Mkdir(dir); err != nil {
			return err
		}
	}

	defer b.cleanup(a1)

	return b.moveOrCopyFile(a, a.Target, a1.built, perm, false)
}

// cleanup removes a's object dir to keep the amount of
// on-disk garbage down in a large build. On an operating system
// with aggressive buffering, cleaning incrementally like
// this keeps the intermediate objects from hitting the disk.
func (b *Builder) cleanup(a *Action) {
	if !cfg.BuildWork {
		if cfg.BuildX {
			b.Showcmd("", "rm -r %s", a.Objdir)
		}
		os.RemoveAll(a.Objdir)
	}
}

// moveOrCopyFile is like 'mv src dst' or 'cp src dst'.
func (b *Builder) moveOrCopyFile(a *Action, dst, src string, perm os.FileMode, force bool) error {
	if cfg.BuildN {
		b.Showcmd("", "mv %s %s", src, dst)
		return nil
	}

	// If we can update the mode and rename to the dst, do it.
	// Otherwise fall back to standard copy.

	// If the source is in the build cache, we need to copy it.
	if strings.HasPrefix(src, cache.DefaultDir()) {
		return b.copyFile(a, dst, src, perm, force)
	}

	// On Windows, always copy the file, so that we respect the NTFS
	// permissions of the parent folder. https://golang.org/issue/22343.
	// What matters here is not cfg.Goos (the system we are building
	// for) but runtime.GOOS (the system we are building on).
	if runtime.GOOS == "windows" {
		return b.copyFile(a, dst, src, perm, force)
	}

	// If the destination directory has the group sticky bit set,
	// we have to copy the file to retain the correct permissions.
	// https://golang.org/issue/18878
	if fi, err := os.Stat(filepath.Dir(dst)); err == nil {
		if fi.IsDir() && (fi.Mode()&os.ModeSetgid) != 0 {
			return b.copyFile(a, dst, src, perm, force)
		}
	}

	// The perm argument is meant to be adjusted according to umask,
	// but we don't know what the umask is.
	// Create a dummy file to find out.
	// This avoids build tags and works even on systems like Plan 9
	// where the file mask computation incorporates other information.
	mode := perm
	f, err := os.OpenFile(filepath.Clean(dst)+"-go-tmp-umask", os.O_WRONLY|os.O_CREATE|os.O_EXCL, perm)
	if err == nil {
		fi, err := f.Stat()
		if err == nil {
			mode = fi.Mode() & 0777
		}
		name := f.Name()
		f.Close()
		os.Remove(name)
	}

	if err := os.Chmod(src, mode); err == nil {
		if err := os.Rename(src, dst); err == nil {
			if cfg.BuildX {
				b.Showcmd("", "mv %s %s", src, dst)
			}
			return nil
		}
	}

	return b.copyFile(a, dst, src, perm, force)
}

// copyFile is like 'cp src dst'.
func (b *Builder) copyFile(a *Action, dst, src string, perm os.FileMode, force bool) error {
	if cfg.BuildN || cfg.BuildX {
		b.Showcmd("", "cp %s %s", src, dst)
		if cfg.BuildN {
			return nil
		}
	}

	sf, err := os.Open(src)
	if err != nil {
		return err
	}
	defer sf.Close()

	// Be careful about removing/overwriting dst.
	// Do not remove/overwrite if dst exists and is a directory
	// or a non-object file.
	if fi, err := os.Stat(dst); err == nil {
		if fi.IsDir() {
			return fmt.Errorf("build output %q already exists and is a directory", dst)
		}
		if !force && fi.Mode().IsRegular() && !isObject(dst) {
			return fmt.Errorf("build output %q already exists and is not an object file", dst)
		}
	}

	// On Windows, remove lingering ~ file from last attempt.
	if base.ToolIsWindows {
		if _, err := os.Stat(dst + "~"); err == nil {
			os.Remove(dst + "~")
		}
	}

	mayberemovefile(dst)
	df, err := os.OpenFile(dst, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, perm)
	if err != nil && base.ToolIsWindows {
		// Windows does not allow deletion of a binary file
		// while it is executing. Try to move it out of the way.
		// If the move fails, which is likely, we'll try again the
		// next time we do an install of this binary.
		if err := os.Rename(dst, dst+"~"); err == nil {
			os.Remove(dst + "~")
		}
		df, err = os.OpenFile(dst, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, perm)
	}
	if err != nil {
		return err
	}

	_, err = io.Copy(df, sf)
	df.Close()
	if err != nil {
		mayberemovefile(dst)
		return fmt.Errorf("copying %s to %s: %v", src, dst, err)
	}
	return nil
}

// writeFile writes the text to file.
func (b *Builder) writeFile(file string, text []byte) error {
	if cfg.BuildN || cfg.BuildX {
		b.Showcmd("", "cat >%s << 'EOF' # internal\n%sEOF", file, text)
	}
	if cfg.BuildN {
		return nil
	}
	return ioutil.WriteFile(file, text, 0666)
}

// Install the cgo export header file, if there is one.
func (b *Builder) installHeader(a *Action) error {
	src := a.Objdir + "_cgo_install.h"
	if _, err := os.Stat(src); os.IsNotExist(err) {
		// If the file does not exist, there are no exported
		// functions, and we do not install anything.
		// TODO(rsc): Once we know that caching is rebuilding
		// at the right times (not missing rebuilds), here we should
		// probably delete the installed header, if any.
		if cfg.BuildX {
			b.Showcmd("", "# %s not created", src)
		}
		return nil
	}

	dir, _ := filepath.Split(a.Target)
	if dir != "" {
		if err := b.Mkdir(dir); err != nil {
			return err
		}
	}

	return b.moveOrCopyFile(a, a.Target, src, 0666, true)
}

// cover runs, in effect,
//	go tool cover -mode=b.coverMode -var="varName" -o dst.go src.go
func (b *Builder) cover(a *Action, dst, src string, perm os.FileMode, varName string) error {
	return b.run(a, a.Objdir, "cover "+a.Package.ImportPath, nil,
		cfg.BuildToolexec,
		base.Tool("cover"),
		"-mode", a.Package.Internal.CoverMode,
		"-var", varName,
		"-o", dst,
		src)
}

var objectMagic = [][]byte{
	{'!', '<', 'a', 'r', 'c', 'h', '>', '\n'}, // Package archive
	{'\x7F', 'E', 'L', 'F'},                   // ELF
	{0xFE, 0xED, 0xFA, 0xCE},                  // Mach-O big-endian 32-bit
	{0xFE, 0xED, 0xFA, 0xCF},                  // Mach-O big-endian 64-bit
	{0xCE, 0xFA, 0xED, 0xFE},                  // Mach-O little-endian 32-bit
	{0xCF, 0xFA, 0xED, 0xFE},                  // Mach-O little-endian 64-bit
	{0x4d, 0x5a, 0x90, 0x00, 0x03, 0x00},      // PE (Windows) as generated by 6l/8l and gcc
	{0x00, 0x00, 0x01, 0xEB},                  // Plan 9 i386
	{0x00, 0x00, 0x8a, 0x97},                  // Plan 9 amd64
	{0x00, 0x00, 0x06, 0x47},                  // Plan 9 arm
}

func isObject(s string) bool {
	f, err := os.Open(s)
	if err != nil {
		return false
	}
	defer f.Close()
	buf := make([]byte, 64)
	io.ReadFull(f, buf)
	for _, magic := range objectMagic {
		if bytes.HasPrefix(buf, magic) {
			return true
		}
	}
	return false
}

// mayberemovefile removes a file only if it is a regular file
// When running as a user with sufficient privileges, we may delete
// even device files, for example, which is not intended.
func mayberemovefile(s string) {
	if fi, err := os.Lstat(s); err == nil && !fi.Mode().IsRegular() {
		return
	}
	os.Remove(s)
}

// fmtcmd formats a command in the manner of fmt.Sprintf but also:
//
//	If dir is non-empty and the script is not in dir right now,
//	fmtcmd inserts "cd dir\n" before the command.
//
//	fmtcmd replaces the value of b.WorkDir with $WORK.
//	fmtcmd replaces the value of goroot with $GOROOT.
//	fmtcmd replaces the value of b.gobin with $GOBIN.
//
//	fmtcmd replaces the name of the current directory with dot (.)
//	but only when it is at the beginning of a space-separated token.
//
func (b *Builder) fmtcmd(dir string, format string, args ...interface{}) string {
	cmd := fmt.Sprintf(format, args...)
	if dir != "" && dir != "/" {
		cmd = strings.Replace(" "+cmd, " "+dir, " .", -1)[1:]
		if b.scriptDir != dir {
			b.scriptDir = dir
			cmd = "cd " + dir + "\n" + cmd
		}
	}
	if b.WorkDir != "" {
		cmd = strings.Replace(cmd, b.WorkDir, "$WORK", -1)
	}
	return cmd
}

// showcmd prints the given command to standard output
// for the implementation of -n or -x.
func (b *Builder) Showcmd(dir string, format string, args ...interface{}) {
	b.output.Lock()
	defer b.output.Unlock()
	b.Print(b.fmtcmd(dir, format, args...) + "\n")
}

// showOutput prints "# desc" followed by the given output.
// The output is expected to contain references to 'dir', usually
// the source directory for the package that has failed to build.
// showOutput rewrites mentions of dir with a relative path to dir
// when the relative path is shorter. This is usually more pleasant.
// For example, if fmt doesn't compile and we are in src/html,
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
//	/usr/gopher/go/src/fmt/print.go:1090: undefined: asdf
//	$
//
// showOutput also replaces references to the work directory with $WORK.
//
// If a is not nil and a.output is not nil, showOutput appends to that slice instead of
// printing to b.Print.
//
func (b *Builder) showOutput(a *Action, dir, desc, out string) {
	prefix := "# " + desc
	suffix := "\n" + out
	if reldir := base.ShortPath(dir); reldir != dir {
		suffix = strings.Replace(suffix, " "+dir, " "+reldir, -1)
		suffix = strings.Replace(suffix, "\n"+dir, "\n"+reldir, -1)
	}
	suffix = strings.Replace(suffix, " "+b.WorkDir, " $WORK", -1)

	if a != nil && a.output != nil {
		a.output = append(a.output, prefix...)
		a.output = append(a.output, suffix...)
		return
	}

	b.output.Lock()
	defer b.output.Unlock()
	b.Print(prefix, suffix)
}

// errPrintedOutput is a special error indicating that a command failed
// but that it generated output as well, and that output has already
// been printed, so there's no point showing 'exit status 1' or whatever
// the wait status was. The main executor, builder.do, knows not to
// print this error.
var errPrintedOutput = errors.New("already printed output - no need to show error")

var cgoLine = regexp.MustCompile(`\[[^\[\]]+\.(cgo1|cover)\.go:[0-9]+(:[0-9]+)?\]`)
var cgoTypeSigRe = regexp.MustCompile(`\b_C2?(type|func|var|macro)_\B`)

// run runs the command given by cmdline in the directory dir.
// If the command fails, run prints information about the failure
// and returns a non-nil error.
func (b *Builder) run(a *Action, dir string, desc string, env []string, cmdargs ...interface{}) error {
	out, err := b.runOut(dir, desc, env, cmdargs...)
	if len(out) > 0 {
		if desc == "" {
			desc = b.fmtcmd(dir, "%s", strings.Join(str.StringList(cmdargs...), " "))
		}
		b.showOutput(a, dir, desc, b.processOutput(out))
		if err != nil {
			err = errPrintedOutput
		}
	}
	return err
}

// processOutput prepares the output of runOut to be output to the console.
func (b *Builder) processOutput(out []byte) string {
	if out[len(out)-1] != '\n' {
		out = append(out, '\n')
	}
	messages := string(out)
	// Fix up output referring to cgo-generated code to be more readable.
	// Replace x.go:19[/tmp/.../x.cgo1.go:18] with x.go:19.
	// Replace *[100]_Ctype_foo with *[100]C.foo.
	// If we're using -x, assume we're debugging and want the full dump, so disable the rewrite.
	if !cfg.BuildX && cgoLine.MatchString(messages) {
		messages = cgoLine.ReplaceAllString(messages, "")
		messages = cgoTypeSigRe.ReplaceAllString(messages, "C.")
	}
	return messages
}

// runOut runs the command given by cmdline in the directory dir.
// It returns the command output and any errors that occurred.
func (b *Builder) runOut(dir string, desc string, env []string, cmdargs ...interface{}) ([]byte, error) {
	cmdline := str.StringList(cmdargs...)

	for _, arg := range cmdline {
		// GNU binutils commands, including gcc and gccgo, interpret an argument
		// @foo anywhere in the command line (even following --) as meaning
		// "read and insert arguments from the file named foo."
		// Don't say anything that might be misinterpreted that way.
		if strings.HasPrefix(arg, "@") {
			return nil, fmt.Errorf("invalid command-line argument %s in command: %s", arg, joinUnambiguously(cmdline))
		}
	}

	if cfg.BuildN || cfg.BuildX {
		var envcmdline string
		for _, e := range env {
			if j := strings.IndexByte(e, '='); j != -1 {
				if strings.ContainsRune(e[j+1:], '\'') {
					envcmdline += fmt.Sprintf("%s=%q", e[:j], e[j+1:])
				} else {
					envcmdline += fmt.Sprintf("%s='%s'", e[:j], e[j+1:])
				}
				envcmdline += " "
			}
		}
		envcmdline += joinUnambiguously(cmdline)
		b.Showcmd(dir, "%s", envcmdline)
		if cfg.BuildN {
			return nil, nil
		}
	}

	var buf bytes.Buffer
	cmd := exec.Command(cmdline[0], cmdline[1:]...)
	cmd.Stdout = &buf
	cmd.Stderr = &buf
	cmd.Dir = dir
	cmd.Env = base.MergeEnvLists(env, base.EnvForDir(cmd.Dir, os.Environ()))
	err := cmd.Run()

	// err can be something like 'exit status 1'.
	// Add information about what program was running.
	// Note that if buf.Bytes() is non-empty, the caller usually
	// shows buf.Bytes() and does not print err at all, so the
	// prefix here does not make most output any more verbose.
	if err != nil {
		err = errors.New(cmdline[0] + ": " + err.Error())
	}
	return buf.Bytes(), err
}

// joinUnambiguously prints the slice, quoting where necessary to make the
// output unambiguous.
// TODO: See issue 5279. The printing of commands needs a complete redo.
func joinUnambiguously(a []string) string {
	var buf bytes.Buffer
	for i, s := range a {
		if i > 0 {
			buf.WriteByte(' ')
		}
		q := strconv.Quote(s)
		if s == "" || strings.Contains(s, " ") || len(q) > len(s)+2 {
			buf.WriteString(q)
		} else {
			buf.WriteString(s)
		}
	}
	return buf.String()
}

// mkdir makes the named directory.
func (b *Builder) Mkdir(dir string) error {
	// Make Mkdir(a.Objdir) a no-op instead of an error when a.Objdir == "".
	if dir == "" {
		return nil
	}

	b.exec.Lock()
	defer b.exec.Unlock()
	// We can be a little aggressive about being
	// sure directories exist. Skip repeated calls.
	if b.mkdirCache[dir] {
		return nil
	}
	b.mkdirCache[dir] = true

	if cfg.BuildN || cfg.BuildX {
		b.Showcmd("", "mkdir -p %s", dir)
		if cfg.BuildN {
			return nil
		}
	}

	if err := os.MkdirAll(dir, 0777); err != nil {
		return err
	}
	return nil
}

// symlink creates a symlink newname -> oldname.
func (b *Builder) Symlink(oldname, newname string) error {
	if cfg.BuildN || cfg.BuildX {
		b.Showcmd("", "ln -s %s %s", oldname, newname)
		if cfg.BuildN {
			return nil
		}
	}
	return os.Symlink(oldname, newname)
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

type toolchain interface {
	// gc runs the compiler in a specific directory on a set of files
	// and returns the name of the generated output file.
	gc(b *Builder, a *Action, archive string, importcfg []byte, asmhdr bool, gofiles []string) (ofile string, out []byte, err error)
	// cc runs the toolchain's C compiler in a directory on a C file
	// to produce an output file.
	cc(b *Builder, a *Action, ofile, cfile string) error
	// asm runs the assembler in a specific directory on specific files
	// and returns a list of named output files.
	asm(b *Builder, a *Action, sfiles []string) ([]string, error)
	// pack runs the archive packer in a specific directory to create
	// an archive from a set of object files.
	// typically it is run in the object directory.
	pack(b *Builder, a *Action, afile string, ofiles []string) error
	// ld runs the linker to create an executable starting at mainpkg.
	ld(b *Builder, root *Action, out, importcfg, mainpkg string) error
	// ldShared runs the linker to create a shared library containing the pkgs built by toplevelactions
	ldShared(b *Builder, root *Action, toplevelactions []*Action, out, importcfg string, allactions []*Action) error

	compiler() string
	linker() string
}

type noToolchain struct{}

func noCompiler() error {
	log.Fatalf("unknown compiler %q", cfg.BuildContext.Compiler)
	return nil
}

func (noToolchain) compiler() string {
	noCompiler()
	return ""
}

func (noToolchain) linker() string {
	noCompiler()
	return ""
}

func (noToolchain) gc(b *Builder, a *Action, archive string, importcfg []byte, asmhdr bool, gofiles []string) (ofile string, out []byte, err error) {
	return "", nil, noCompiler()
}

func (noToolchain) asm(b *Builder, a *Action, sfiles []string) ([]string, error) {
	return nil, noCompiler()
}

func (noToolchain) pack(b *Builder, a *Action, afile string, ofiles []string) error {
	return noCompiler()
}

func (noToolchain) ld(b *Builder, root *Action, out, importcfg, mainpkg string) error {
	return noCompiler()
}

func (noToolchain) ldShared(b *Builder, root *Action, toplevelactions []*Action, out, importcfg string, allactions []*Action) error {
	return noCompiler()
}

func (noToolchain) cc(b *Builder, a *Action, ofile, cfile string) error {
	return noCompiler()
}

// gcc runs the gcc C compiler to create an object from a single C file.
func (b *Builder) gcc(a *Action, p *load.Package, workdir, out string, flags []string, cfile string) error {
	return b.ccompile(a, p, out, flags, cfile, b.GccCmd(p.Dir, workdir))
}

// gxx runs the g++ C++ compiler to create an object from a single C++ file.
func (b *Builder) gxx(a *Action, p *load.Package, workdir, out string, flags []string, cxxfile string) error {
	return b.ccompile(a, p, out, flags, cxxfile, b.GxxCmd(p.Dir, workdir))
}

// gfortran runs the gfortran Fortran compiler to create an object from a single Fortran file.
func (b *Builder) gfortran(a *Action, p *load.Package, workdir, out string, flags []string, ffile string) error {
	return b.ccompile(a, p, out, flags, ffile, b.gfortranCmd(p.Dir, workdir))
}

// ccompile runs the given C or C++ compiler and creates an object from a single source file.
func (b *Builder) ccompile(a *Action, p *load.Package, outfile string, flags []string, file string, compiler []string) error {
	file = mkAbs(p.Dir, file)
	desc := p.ImportPath
	if !filepath.IsAbs(outfile) {
		outfile = filepath.Join(p.Dir, outfile)
	}
	output, err := b.runOut(filepath.Dir(file), desc, nil, compiler, flags, "-o", outfile, "-c", filepath.Base(file))
	if len(output) > 0 {
		// On FreeBSD 11, when we pass -g to clang 3.8 it
		// invokes its internal assembler with -dwarf-version=2.
		// When it sees .section .note.GNU-stack, it warns
		// "DWARF2 only supports one section per compilation unit".
		// This warning makes no sense, since the section is empty,
		// but it confuses people.
		// We work around the problem by detecting the warning
		// and dropping -g and trying again.
		if bytes.Contains(output, []byte("DWARF2 only supports one section per compilation unit")) {
			newFlags := make([]string, 0, len(flags))
			for _, f := range flags {
				if !strings.HasPrefix(f, "-g") {
					newFlags = append(newFlags, f)
				}
			}
			if len(newFlags) < len(flags) {
				return b.ccompile(a, p, outfile, newFlags, file, compiler)
			}
		}

		b.showOutput(a, p.Dir, desc, b.processOutput(output))
		if err != nil {
			err = errPrintedOutput
		} else if os.Getenv("GO_BUILDER_NAME") != "" {
			return errors.New("C compiler warning promoted to error on Go builders")
		}
	}
	return err
}

// gccld runs the gcc linker to create an executable from a set of object files.
func (b *Builder) gccld(p *load.Package, objdir, out string, flags []string, objs []string) error {
	var cmd []string
	if len(p.CXXFiles) > 0 || len(p.SwigCXXFiles) > 0 {
		cmd = b.GxxCmd(p.Dir, objdir)
	} else {
		cmd = b.GccCmd(p.Dir, objdir)
	}
	return b.run(nil, p.Dir, p.ImportPath, nil, cmd, "-o", out, objs, flags)
}

// Grab these before main helpfully overwrites them.
var (
	origCC  = os.Getenv("CC")
	origCXX = os.Getenv("CXX")
)

// gccCmd returns a gcc command line prefix
// defaultCC is defined in zdefaultcc.go, written by cmd/dist.
func (b *Builder) GccCmd(incdir, workdir string) []string {
	return b.compilerCmd(b.ccExe(), incdir, workdir)
}

// gxxCmd returns a g++ command line prefix
// defaultCXX is defined in zdefaultcc.go, written by cmd/dist.
func (b *Builder) GxxCmd(incdir, workdir string) []string {
	return b.compilerCmd(b.cxxExe(), incdir, workdir)
}

// gfortranCmd returns a gfortran command line prefix.
func (b *Builder) gfortranCmd(incdir, workdir string) []string {
	return b.compilerCmd(b.fcExe(), incdir, workdir)
}

// ccExe returns the CC compiler setting without all the extra flags we add implicitly.
func (b *Builder) ccExe() []string {
	return b.compilerExe(origCC, cfg.DefaultCC(cfg.Goos, cfg.Goarch))
}

// cxxExe returns the CXX compiler setting without all the extra flags we add implicitly.
func (b *Builder) cxxExe() []string {
	return b.compilerExe(origCXX, cfg.DefaultCXX(cfg.Goos, cfg.Goarch))
}

// fcExe returns the FC compiler setting without all the extra flags we add implicitly.
func (b *Builder) fcExe() []string {
	return b.compilerExe(os.Getenv("FC"), "gfortran")
}

// compilerExe returns the compiler to use given an
// environment variable setting (the value not the name)
// and a default. The resulting slice is usually just the name
// of the compiler but can have additional arguments if they
// were present in the environment value.
// For example if CC="gcc -DGOPHER" then the result is ["gcc", "-DGOPHER"].
func (b *Builder) compilerExe(envValue string, def string) []string {
	compiler := strings.Fields(envValue)
	if len(compiler) == 0 {
		compiler = []string{def}
	}
	return compiler
}

// compilerCmd returns a command line prefix for the given environment
// variable and using the default command when the variable is empty.
func (b *Builder) compilerCmd(compiler []string, incdir, workdir string) []string {
	// NOTE: env.go's mkEnv knows that the first three
	// strings returned are "gcc", "-I", incdir (and cuts them off).
	a := []string{compiler[0], "-I", incdir}
	a = append(a, compiler[1:]...)

	// Definitely want -fPIC but on Windows gcc complains
	// "-fPIC ignored for target (all code is position independent)"
	if cfg.Goos != "windows" {
		a = append(a, "-fPIC")
	}
	a = append(a, b.gccArchArgs()...)
	// gcc-4.5 and beyond require explicit "-pthread" flag
	// for multithreading with pthread library.
	if cfg.BuildContext.CgoEnabled {
		switch cfg.Goos {
		case "windows":
			a = append(a, "-mthreads")
		default:
			a = append(a, "-pthread")
		}
	}

	// disable ASCII art in clang errors, if possible
	if b.gccSupportsFlag(compiler, "-fno-caret-diagnostics") {
		a = append(a, "-fno-caret-diagnostics")
	}
	// clang is too smart about command-line arguments
	if b.gccSupportsFlag(compiler, "-Qunused-arguments") {
		a = append(a, "-Qunused-arguments")
	}

	// disable word wrapping in error messages
	a = append(a, "-fmessage-length=0")

	// Tell gcc not to include the work directory in object files.
	if b.gccSupportsFlag(compiler, "-fdebug-prefix-map=a=b") {
		if workdir == "" {
			workdir = b.WorkDir
		}
		workdir = strings.TrimSuffix(workdir, string(filepath.Separator))
		a = append(a, "-fdebug-prefix-map="+workdir+"=/tmp/go-build")
	}

	// Tell gcc not to include flags in object files, which defeats the
	// point of -fdebug-prefix-map above.
	if b.gccSupportsFlag(compiler, "-gno-record-gcc-switches") {
		a = append(a, "-gno-record-gcc-switches")
	}

	// On OS X, some of the compilers behave as if -fno-common
	// is always set, and the Mach-O linker in 6l/8l assumes this.
	// See https://golang.org/issue/3253.
	if cfg.Goos == "darwin" {
		a = append(a, "-fno-common")
	}

	return a
}

// gccNoPie returns the flag to use to request non-PIE. On systems
// with PIE (position independent executables) enabled by default,
// -no-pie must be passed when doing a partial link with -Wl,-r.
// But -no-pie is not supported by all compilers, and clang spells it -nopie.
func (b *Builder) gccNoPie(linker []string) string {
	if b.gccSupportsFlag(linker, "-no-pie") {
		return "-no-pie"
	}
	if b.gccSupportsFlag(linker, "-nopie") {
		return "-nopie"
	}
	return ""
}

// gccSupportsFlag checks to see if the compiler supports a flag.
func (b *Builder) gccSupportsFlag(compiler []string, flag string) bool {
	key := [2]string{compiler[0], flag}

	b.exec.Lock()
	defer b.exec.Unlock()
	if b, ok := b.flagCache[key]; ok {
		return b
	}
	if b.flagCache == nil {
		b.flagCache = make(map[[2]string]bool)
	}
	// We used to write an empty C file, but that gets complicated with
	// go build -n. We tried using a file that does not exist, but that
	// fails on systems with GCC version 4.2.1; that is the last GPLv2
	// version of GCC, so some systems have frozen on it.
	// Now we pass an empty file on stdin, which should work at least for
	// GCC and clang.
	cmdArgs := str.StringList(compiler, flag, "-c", "-x", "c", "-")
	if cfg.BuildN || cfg.BuildX {
		b.Showcmd(b.WorkDir, "%s || true", joinUnambiguously(cmdArgs))
		if cfg.BuildN {
			return false
		}
	}
	cmd := exec.Command(cmdArgs[0], cmdArgs[1:]...)
	cmd.Dir = b.WorkDir
	cmd.Env = base.MergeEnvLists([]string{"LC_ALL=C"}, base.EnvForDir(cmd.Dir, os.Environ()))
	out, _ := cmd.CombinedOutput()
	// GCC says "unrecognized command line option".
	// clang says "unknown argument".
	// Older versions of GCC say "unrecognised debug output level".
	// For -fsplit-stack GCC says "'-fsplit-stack' is not supported".
	supported := !bytes.Contains(out, []byte("unrecognized")) &&
		!bytes.Contains(out, []byte("unknown")) &&
		!bytes.Contains(out, []byte("unrecognised")) &&
		!bytes.Contains(out, []byte("is not supported"))
	b.flagCache[key] = supported
	return supported
}

// gccArchArgs returns arguments to pass to gcc based on the architecture.
func (b *Builder) gccArchArgs() []string {
	switch cfg.Goarch {
	case "386":
		return []string{"-m32"}
	case "amd64", "amd64p32":
		return []string{"-m64"}
	case "arm":
		return []string{"-marm"} // not thumb
	case "s390x":
		return []string{"-m64", "-march=z196"}
	case "mips64", "mips64le":
		return []string{"-mabi=64"}
	case "mips", "mipsle":
		return []string{"-mabi=32", "-march=mips32"}
	}
	return nil
}

// envList returns the value of the given environment variable broken
// into fields, using the default value when the variable is empty.
func envList(key, def string) []string {
	v := os.Getenv(key)
	if v == "" {
		v = def
	}
	return strings.Fields(v)
}

// CFlags returns the flags to use when invoking the C, C++ or Fortran compilers, or cgo.
func (b *Builder) CFlags(p *load.Package) (cppflags, cflags, cxxflags, fflags, ldflags []string, err error) {
	defaults := "-g -O2"

	if cppflags, err = buildFlags("CPPFLAGS", "", p.CgoCPPFLAGS, checkCompilerFlags); err != nil {
		return
	}
	if cflags, err = buildFlags("CFLAGS", defaults, p.CgoCFLAGS, checkCompilerFlags); err != nil {
		return
	}
	if cxxflags, err = buildFlags("CXXFLAGS", defaults, p.CgoCXXFLAGS, checkCompilerFlags); err != nil {
		return
	}
	if fflags, err = buildFlags("FFLAGS", defaults, p.CgoFFLAGS, checkCompilerFlags); err != nil {
		return
	}
	if ldflags, err = buildFlags("LDFLAGS", defaults, p.CgoLDFLAGS, checkLinkerFlags); err != nil {
		return
	}

	return
}

func buildFlags(name, defaults string, fromPackage []string, check func(string, string, []string) error) ([]string, error) {
	if err := check(name, "#cgo "+name, fromPackage); err != nil {
		return nil, err
	}
	return str.StringList(envList("CGO_"+name, defaults), fromPackage), nil
}

var cgoRe = regexp.MustCompile(`[/\\:]`)

func (b *Builder) cgo(a *Action, cgoExe, objdir string, pcCFLAGS, pcLDFLAGS, cgofiles, gccfiles, gxxfiles, mfiles, ffiles []string) (outGo, outObj []string, err error) {
	p := a.Package
	cgoCPPFLAGS, cgoCFLAGS, cgoCXXFLAGS, cgoFFLAGS, cgoLDFLAGS, err := b.CFlags(p)
	if err != nil {
		return nil, nil, err
	}

	cgoCPPFLAGS = append(cgoCPPFLAGS, pcCFLAGS...)
	cgoLDFLAGS = append(cgoLDFLAGS, pcLDFLAGS...)
	// If we are compiling Objective-C code, then we need to link against libobjc
	if len(mfiles) > 0 {
		cgoLDFLAGS = append(cgoLDFLAGS, "-lobjc")
	}

	// Likewise for Fortran, except there are many Fortran compilers.
	// Support gfortran out of the box and let others pass the correct link options
	// via CGO_LDFLAGS
	if len(ffiles) > 0 {
		fc := os.Getenv("FC")
		if fc == "" {
			fc = "gfortran"
		}
		if strings.Contains(fc, "gfortran") {
			cgoLDFLAGS = append(cgoLDFLAGS, "-lgfortran")
		}
	}

	if cfg.BuildMSan {
		cgoCFLAGS = append([]string{"-fsanitize=memory"}, cgoCFLAGS...)
		cgoLDFLAGS = append([]string{"-fsanitize=memory"}, cgoLDFLAGS...)
	}

	// Allows including _cgo_export.h from .[ch] files in the package.
	cgoCPPFLAGS = append(cgoCPPFLAGS, "-I", objdir)

	// cgo
	// TODO: CGO_FLAGS?
	gofiles := []string{objdir + "_cgo_gotypes.go"}
	cfiles := []string{"_cgo_export.c"}
	for _, fn := range cgofiles {
		f := strings.TrimSuffix(filepath.Base(fn), ".go")
		gofiles = append(gofiles, objdir+f+".cgo1.go")
		cfiles = append(cfiles, f+".cgo2.c")
	}

	// TODO: make cgo not depend on $GOARCH?

	cgoflags := []string{}
	if p.Standard && p.ImportPath == "runtime/cgo" {
		cgoflags = append(cgoflags, "-import_runtime_cgo=false")
	}
	if p.Standard && (p.ImportPath == "runtime/race" || p.ImportPath == "runtime/msan" || p.ImportPath == "runtime/cgo") {
		cgoflags = append(cgoflags, "-import_syscall=false")
	}

	// Update $CGO_LDFLAGS with p.CgoLDFLAGS.
	// These flags are recorded in the generated _cgo_gotypes.go file
	// using //go:cgo_ldflag directives, the compiler records them in the
	// object file for the package, and then the Go linker passes them
	// along to the host linker. At this point in the code, cgoLDFLAGS
	// consists of the original $CGO_LDFLAGS (unchecked) and all the
	// flags put together from source code (checked).
	var cgoenv []string
	if len(cgoLDFLAGS) > 0 {
		flags := make([]string, len(cgoLDFLAGS))
		for i, f := range cgoLDFLAGS {
			flags[i] = strconv.Quote(f)
		}
		cgoenv = []string{"CGO_LDFLAGS=" + strings.Join(flags, " ")}
	}

	if cfg.BuildToolchainName == "gccgo" {
		switch cfg.Goarch {
		case "386", "amd64":
			cgoCFLAGS = append(cgoCFLAGS, "-fsplit-stack")
		}
		cgoflags = append(cgoflags, "-gccgo")
		if pkgpath := gccgoPkgpath(p); pkgpath != "" {
			cgoflags = append(cgoflags, "-gccgopkgpath="+pkgpath)
		}
	}

	switch cfg.BuildBuildmode {
	case "c-archive", "c-shared":
		// Tell cgo that if there are any exported functions
		// it should generate a header file that C code can
		// #include.
		cgoflags = append(cgoflags, "-exportheader="+objdir+"_cgo_install.h")
	}

	if err := b.run(a, p.Dir, p.ImportPath, cgoenv, cfg.BuildToolexec, cgoExe, "-objdir", objdir, "-importpath", p.ImportPath, cgoflags, "--", cgoCPPFLAGS, cgoCFLAGS, cgofiles); err != nil {
		return nil, nil, err
	}
	outGo = append(outGo, gofiles...)

	// Use sequential object file names to keep them distinct
	// and short enough to fit in the .a header file name slots.
	// We no longer collect them all into _all.o, and we'd like
	// tools to see both the .o suffix and unique names, so
	// we need to make them short enough not to be truncated
	// in the final archive.
	oseq := 0
	nextOfile := func() string {
		oseq++
		return objdir + fmt.Sprintf("_x%03d.o", oseq)
	}

	// gcc
	cflags := str.StringList(cgoCPPFLAGS, cgoCFLAGS)
	for _, cfile := range cfiles {
		ofile := nextOfile()
		if err := b.gcc(a, p, a.Objdir, ofile, cflags, objdir+cfile); err != nil {
			return nil, nil, err
		}
		outObj = append(outObj, ofile)
	}

	for _, file := range gccfiles {
		ofile := nextOfile()
		if err := b.gcc(a, p, a.Objdir, ofile, cflags, file); err != nil {
			return nil, nil, err
		}
		outObj = append(outObj, ofile)
	}

	cxxflags := str.StringList(cgoCPPFLAGS, cgoCXXFLAGS)
	for _, file := range gxxfiles {
		ofile := nextOfile()
		if err := b.gxx(a, p, a.Objdir, ofile, cxxflags, file); err != nil {
			return nil, nil, err
		}
		outObj = append(outObj, ofile)
	}

	for _, file := range mfiles {
		ofile := nextOfile()
		if err := b.gcc(a, p, a.Objdir, ofile, cflags, file); err != nil {
			return nil, nil, err
		}
		outObj = append(outObj, ofile)
	}

	fflags := str.StringList(cgoCPPFLAGS, cgoFFLAGS)
	for _, file := range ffiles {
		ofile := nextOfile()
		if err := b.gfortran(a, p, a.Objdir, ofile, fflags, file); err != nil {
			return nil, nil, err
		}
		outObj = append(outObj, ofile)
	}

	switch cfg.BuildToolchainName {
	case "gc":
		importGo := objdir + "_cgo_import.go"
		if err := b.dynimport(a, p, objdir, importGo, cgoExe, cflags, cgoLDFLAGS, outObj); err != nil {
			return nil, nil, err
		}
		outGo = append(outGo, importGo)

	case "gccgo":
		defunC := objdir + "_cgo_defun.c"
		defunObj := objdir + "_cgo_defun.o"
		if err := BuildToolchain.cc(b, a, defunObj, defunC); err != nil {
			return nil, nil, err
		}
		outObj = append(outObj, defunObj)

	default:
		noCompiler()
	}

	return outGo, outObj, nil
}

// dynimport creates a Go source file named importGo containing
// //go:cgo_import_dynamic directives for each symbol or library
// dynamically imported by the object files outObj.
func (b *Builder) dynimport(a *Action, p *load.Package, objdir, importGo, cgoExe string, cflags, cgoLDFLAGS, outObj []string) error {
	cfile := objdir + "_cgo_main.c"
	ofile := objdir + "_cgo_main.o"
	if err := b.gcc(a, p, objdir, ofile, cflags, cfile); err != nil {
		return err
	}

	linkobj := str.StringList(ofile, outObj, p.SysoFiles)
	dynobj := objdir + "_cgo_.o"

	// we need to use -pie for Linux/ARM to get accurate imported sym
	ldflags := cgoLDFLAGS
	if (cfg.Goarch == "arm" && cfg.Goos == "linux") || cfg.Goos == "android" {
		ldflags = append(ldflags, "-pie")
	}
	if err := b.gccld(p, objdir, dynobj, ldflags, linkobj); err != nil {
		return err
	}

	// cgo -dynimport
	var cgoflags []string
	if p.Standard && p.ImportPath == "runtime/cgo" {
		cgoflags = []string{"-dynlinker"} // record path to dynamic linker
	}
	return b.run(a, p.Dir, p.ImportPath, nil, cfg.BuildToolexec, cgoExe, "-dynpackage", p.Name, "-dynimport", dynobj, "-dynout", importGo, cgoflags)
}

// Run SWIG on all SWIG input files.
// TODO: Don't build a shared library, once SWIG emits the necessary
// pragmas for external linking.
func (b *Builder) swig(a *Action, p *load.Package, objdir string, pcCFLAGS []string) (outGo, outC, outCXX []string, err error) {
	if err := b.swigVersionCheck(); err != nil {
		return nil, nil, nil, err
	}

	intgosize, err := b.swigIntSize(objdir)
	if err != nil {
		return nil, nil, nil, err
	}

	for _, f := range p.SwigFiles {
		goFile, cFile, err := b.swigOne(a, p, f, objdir, pcCFLAGS, false, intgosize)
		if err != nil {
			return nil, nil, nil, err
		}
		if goFile != "" {
			outGo = append(outGo, goFile)
		}
		if cFile != "" {
			outC = append(outC, cFile)
		}
	}
	for _, f := range p.SwigCXXFiles {
		goFile, cxxFile, err := b.swigOne(a, p, f, objdir, pcCFLAGS, true, intgosize)
		if err != nil {
			return nil, nil, nil, err
		}
		if goFile != "" {
			outGo = append(outGo, goFile)
		}
		if cxxFile != "" {
			outCXX = append(outCXX, cxxFile)
		}
	}
	return outGo, outC, outCXX, nil
}

// Make sure SWIG is new enough.
var (
	swigCheckOnce sync.Once
	swigCheck     error
)

func (b *Builder) swigDoVersionCheck() error {
	out, err := b.runOut("", "", nil, "swig", "-version")
	if err != nil {
		return err
	}
	re := regexp.MustCompile(`[vV]ersion +([\d]+)([.][\d]+)?([.][\d]+)?`)
	matches := re.FindSubmatch(out)
	if matches == nil {
		// Can't find version number; hope for the best.
		return nil
	}

	major, err := strconv.Atoi(string(matches[1]))
	if err != nil {
		// Can't find version number; hope for the best.
		return nil
	}
	const errmsg = "must have SWIG version >= 3.0.6"
	if major < 3 {
		return errors.New(errmsg)
	}
	if major > 3 {
		// 4.0 or later
		return nil
	}

	// We have SWIG version 3.x.
	if len(matches[2]) > 0 {
		minor, err := strconv.Atoi(string(matches[2][1:]))
		if err != nil {
			return nil
		}
		if minor > 0 {
			// 3.1 or later
			return nil
		}
	}

	// We have SWIG version 3.0.x.
	if len(matches[3]) > 0 {
		patch, err := strconv.Atoi(string(matches[3][1:]))
		if err != nil {
			return nil
		}
		if patch < 6 {
			// Before 3.0.6.
			return errors.New(errmsg)
		}
	}

	return nil
}

func (b *Builder) swigVersionCheck() error {
	swigCheckOnce.Do(func() {
		swigCheck = b.swigDoVersionCheck()
	})
	return swigCheck
}

// Find the value to pass for the -intgosize option to swig.
var (
	swigIntSizeOnce  sync.Once
	swigIntSize      string
	swigIntSizeError error
)

// This code fails to build if sizeof(int) <= 32
const swigIntSizeCode = `
package main
const i int = 1 << 32
`

// Determine the size of int on the target system for the -intgosize option
// of swig >= 2.0.9. Run only once.
func (b *Builder) swigDoIntSize(objdir string) (intsize string, err error) {
	if cfg.BuildN {
		return "$INTBITS", nil
	}
	src := filepath.Join(b.WorkDir, "swig_intsize.go")
	if err = ioutil.WriteFile(src, []byte(swigIntSizeCode), 0666); err != nil {
		return
	}
	srcs := []string{src}

	p := load.GoFilesPackage(srcs)

	if _, _, e := BuildToolchain.gc(b, &Action{Mode: "swigDoIntSize", Package: p, Objdir: objdir}, "", nil, false, srcs); e != nil {
		return "32", nil
	}
	return "64", nil
}

// Determine the size of int on the target system for the -intgosize option
// of swig >= 2.0.9.
func (b *Builder) swigIntSize(objdir string) (intsize string, err error) {
	swigIntSizeOnce.Do(func() {
		swigIntSize, swigIntSizeError = b.swigDoIntSize(objdir)
	})
	return swigIntSize, swigIntSizeError
}

// Run SWIG on one SWIG input file.
func (b *Builder) swigOne(a *Action, p *load.Package, file, objdir string, pcCFLAGS []string, cxx bool, intgosize string) (outGo, outC string, err error) {
	cgoCPPFLAGS, cgoCFLAGS, cgoCXXFLAGS, _, _, err := b.CFlags(p)
	if err != nil {
		return "", "", err
	}

	var cflags []string
	if cxx {
		cflags = str.StringList(cgoCPPFLAGS, pcCFLAGS, cgoCXXFLAGS)
	} else {
		cflags = str.StringList(cgoCPPFLAGS, pcCFLAGS, cgoCFLAGS)
	}

	n := 5 // length of ".swig"
	if cxx {
		n = 8 // length of ".swigcxx"
	}
	base := file[:len(file)-n]
	goFile := base + ".go"
	gccBase := base + "_wrap."
	gccExt := "c"
	if cxx {
		gccExt = "cxx"
	}

	gccgo := cfg.BuildToolchainName == "gccgo"

	// swig
	args := []string{
		"-go",
		"-cgo",
		"-intgosize", intgosize,
		"-module", base,
		"-o", objdir + gccBase + gccExt,
		"-outdir", objdir,
	}

	for _, f := range cflags {
		if len(f) > 3 && f[:2] == "-I" {
			args = append(args, f)
		}
	}

	if gccgo {
		args = append(args, "-gccgo")
		if pkgpath := gccgoPkgpath(p); pkgpath != "" {
			args = append(args, "-go-pkgpath", pkgpath)
		}
	}
	if cxx {
		args = append(args, "-c++")
	}

	out, err := b.runOut(p.Dir, p.ImportPath, nil, "swig", args, file)
	if err != nil {
		if len(out) > 0 {
			if bytes.Contains(out, []byte("-intgosize")) || bytes.Contains(out, []byte("-cgo")) {
				return "", "", errors.New("must have SWIG version >= 3.0.6")
			}
			b.showOutput(a, p.Dir, p.ImportPath, b.processOutput(out)) // swig error
			return "", "", errPrintedOutput
		}
		return "", "", err
	}
	if len(out) > 0 {
		b.showOutput(a, p.Dir, p.ImportPath, b.processOutput(out)) // swig warning
	}

	// If the input was x.swig, the output is x.go in the objdir.
	// But there might be an x.go in the original dir too, and if it
	// uses cgo as well, cgo will be processing both and will
	// translate both into x.cgo1.go in the objdir, overwriting one.
	// Rename x.go to _x_swig.go to avoid this problem.
	// We ignore files in the original dir that begin with underscore
	// so _x_swig.go cannot conflict with an original file we were
	// going to compile.
	goFile = objdir + goFile
	newGoFile := objdir + "_" + base + "_swig.go"
	if err := os.Rename(goFile, newGoFile); err != nil {
		return "", "", err
	}
	return newGoFile, objdir + gccBase + gccExt, nil
}

// disableBuildID adjusts a linker command line to avoid creating a
// build ID when creating an object file rather than an executable or
// shared library. Some systems, such as Ubuntu, always add
// --build-id to every link, but we don't want a build ID when we are
// producing an object file. On some of those system a plain -r (not
// -Wl,-r) will turn off --build-id, but clang 3.0 doesn't support a
// plain -r. I don't know how to turn off --build-id when using clang
// other than passing a trailing --build-id=none. So that is what we
// do, but only on systems likely to support it, which is to say,
// systems that normally use gold or the GNU linker.
func (b *Builder) disableBuildID(ldflags []string) []string {
	switch cfg.Goos {
	case "android", "dragonfly", "linux", "netbsd":
		ldflags = append(ldflags, "-Wl,--build-id=none")
	}
	return ldflags
}

// mkAbsFiles converts files into a list of absolute files,
// assuming they were originally relative to dir,
// and returns that new list.
func mkAbsFiles(dir string, files []string) []string {
	abs := make([]string, len(files))
	for i, f := range files {
		if !filepath.IsAbs(f) {
			f = filepath.Join(dir, f)
		}
		abs[i] = f
	}
	return abs
}
