// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Action graph execution.

package work

import (
	"bytes"
	"cmd/internal/cov/covcmd"
	"cmd/internal/pathcache"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"go/token"
	"internal/lazyregexp"
	"io"
	"io/fs"
	"log"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"slices"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"cmd/go/internal/base"
	"cmd/go/internal/cache"
	"cmd/go/internal/cfg"
	"cmd/go/internal/fsys"
	"cmd/go/internal/gover"
	"cmd/go/internal/load"
	"cmd/go/internal/modload"
	"cmd/go/internal/str"
	"cmd/go/internal/trace"
	"cmd/internal/buildid"
	"cmd/internal/quoted"
	"cmd/internal/sys"
)

const DefaultCFlags = "-O2 -g"

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

// Do runs the action graph rooted at root.
func (b *Builder) Do(ctx context.Context, root *Action) {
	ctx, span := trace.StartSpan(ctx, "exec.Builder.Do ("+root.Mode+" "+root.Target+")")
	defer span.Done()

	if !b.IsCmdList {
		// If we're doing real work, take time at the end to trim the cache.
		c := cache.Default()
		defer func() {
			if err := c.Close(); err != nil {
				base.Fatalf("go: failed to trim cache: %v", err)
			}
		}()
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

	// Write action graph, without timing information, in case we fail and exit early.
	writeActionGraph := func() {
		if file := cfg.DebugActiongraph; file != "" {
			if strings.HasSuffix(file, ".go") {
				// Do not overwrite Go source code in:
				//	go build -debug-actiongraph x.go
				base.Fatalf("go: refusing to write action graph to %v\n", file)
			}
			js := actionGraphJSON(root)
			if err := os.WriteFile(file, []byte(js), 0666); err != nil {
				fmt.Fprintf(os.Stderr, "go: writing action graph: %v\n", err)
				base.SetExitStatus(1)
			}
		}
	}
	writeActionGraph()

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
	handle := func(ctx context.Context, a *Action) {
		if a.json != nil {
			a.json.TimeStart = time.Now()
		}
		var err error
		if a.Actor != nil && (a.Failed == nil || a.IgnoreFail) {
			// TODO(matloob): Better action descriptions
			desc := "Executing action (" + a.Mode
			if a.Package != nil {
				desc += " " + a.Package.Desc()
			}
			desc += ")"
			ctx, span := trace.StartSpan(ctx, desc)
			a.traceSpan = span
			for _, d := range a.Deps {
				trace.Flow(ctx, d.traceSpan, a.traceSpan)
			}
			err = a.Actor.Act(b, ctx, a)
			span.Done()
		}
		if a.json != nil {
			a.json.TimeDone = time.Now()
		}

		// The actions run in parallel but all the updates to the
		// shared work state are serialized through b.exec.
		b.exec.Lock()
		defer b.exec.Unlock()

		if err != nil {
			if b.AllowErrors && a.Package != nil {
				if a.Package.Error == nil {
					a.Package.Error = &load.PackageError{Err: err}
					a.Package.Incomplete = true
				}
			} else {
				var ipe load.ImportPathError
				if a.Package != nil && (!errors.As(err, &ipe) || ipe.ImportPath() != a.Package.ImportPath) {
					err = fmt.Errorf("%s: %v", a.Package.ImportPath, err)
				}
				sh := b.Shell(a)
				sh.Errorf("%s", err)
			}
			if a.Failed == nil {
				a.Failed = a
			}
		}

		for _, a0 := range a.triggers {
			if a.Failed != nil {
				a0.Failed = a.Failed
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
			ctx := trace.StartGoroutine(ctx)
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
					handle(ctx, a)
				case <-base.Interrupted:
					base.SetExitStatus(1)
					return
				}
			}
		}()
	}

	wg.Wait()

	// Write action graph again, this time with timing information.
	writeActionGraph()
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

	// Include information about the origin of the package that
	// may be embedded in the debug info for the object file.
	if cfg.BuildTrimpath {
		// When -trimpath is used with a package built from the module cache,
		// its debug information refers to the module path and version
		// instead of the directory.
		if p.Module != nil {
			fmt.Fprintf(h, "module %s@%s\n", p.Module.Path, p.Module.Version)
		}
	} else if p.Goroot {
		// The Go compiler always hides the exact value of $GOROOT
		// when building things in GOROOT.
		//
		// The C compiler does not, but for packages in GOROOT we rewrite the path
		// as though -trimpath were set. This used to be so that we did not invalidate
		// the build cache (and especially precompiled archive files) when changing
		// GOROOT_FINAL, but we no longer ship precompiled archive files as of Go 1.20
		// (https://go.dev/issue/47257) and no longer support GOROOT_FINAL
		// (https://go.dev/issue/62047).
		// TODO(bcmills): Figure out whether this behavior is still useful.
		//
		// b.WorkDir is always either trimmed or rewritten to
		// the literal string "/tmp/go-build".
	} else if !strings.HasPrefix(p.Dir, b.WorkDir) {
		// -trimpath is not set and no other rewrite rules apply,
		// so the object file may refer to the absolute directory
		// containing the package.
		fmt.Fprintf(h, "dir %s\n", p.Dir)
	}

	if p.Module != nil {
		fmt.Fprintf(h, "go %s\n", p.Module.GoVersion)
	}
	fmt.Fprintf(h, "goos %s goarch %s\n", cfg.Goos, cfg.Goarch)
	fmt.Fprintf(h, "import %q\n", p.ImportPath)
	fmt.Fprintf(h, "omitdebug %v standard %v local %v prefix %q\n", p.Internal.OmitDebug, p.Standard, p.Internal.Local, p.Internal.LocalPrefix)
	if cfg.BuildTrimpath {
		fmt.Fprintln(h, "trimpath")
	}
	if p.Internal.ForceLibrary {
		fmt.Fprintf(h, "forcelibrary\n")
	}
	if len(p.CgoFiles)+len(p.SwigFiles)+len(p.SwigCXXFiles) > 0 {
		fmt.Fprintf(h, "cgo %q\n", b.toolID("cgo"))
		cppflags, cflags, cxxflags, fflags, ldflags, _ := b.CFlags(p)

		ccExe := b.ccExe()
		fmt.Fprintf(h, "CC=%q %q %q %q\n", ccExe, cppflags, cflags, ldflags)
		// Include the C compiler tool ID so that if the C
		// compiler changes we rebuild the package.
		if ccID, _, err := b.gccToolID(ccExe[0], "c"); err == nil {
			fmt.Fprintf(h, "CC ID=%q\n", ccID)
		} else {
			fmt.Fprintf(h, "CC ID ERROR=%q\n", err)
		}
		if len(p.CXXFiles)+len(p.SwigCXXFiles) > 0 {
			cxxExe := b.cxxExe()
			fmt.Fprintf(h, "CXX=%q %q\n", cxxExe, cxxflags)
			if cxxID, _, err := b.gccToolID(cxxExe[0], "c++"); err == nil {
				fmt.Fprintf(h, "CXX ID=%q\n", cxxID)
			} else {
				fmt.Fprintf(h, "CXX ID ERROR=%q\n", err)
			}
		}
		if len(p.FFiles) > 0 {
			fcExe := b.fcExe()
			fmt.Fprintf(h, "FC=%q %q\n", fcExe, fflags)
			if fcID, _, err := b.gccToolID(fcExe[0], "f95"); err == nil {
				fmt.Fprintf(h, "FC ID=%q\n", fcID)
			} else {
				fmt.Fprintf(h, "FC ID ERROR=%q\n", err)
			}
		}
		// TODO(rsc): Should we include the SWIG version?
	}
	if p.Internal.Cover.Mode != "" {
		fmt.Fprintf(h, "cover %q %q\n", p.Internal.Cover.Mode, b.toolID("cover"))
	}
	if p.Internal.FuzzInstrument {
		if fuzzFlags := fuzzInstrumentFlags(); fuzzFlags != nil {
			fmt.Fprintf(h, "fuzz %q\n", fuzzFlags)
		}
	}
	if p.Internal.BuildInfo != nil {
		fmt.Fprintf(h, "modinfo %q\n", p.Internal.BuildInfo.String())
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

		// GOARM, GOMIPS, etc.
		key, val, _ := cfg.GetArchEnv()
		fmt.Fprintf(h, "%s=%s\n", key, val)

		if cfg.CleanGOEXPERIMENT != "" {
			fmt.Fprintf(h, "GOEXPERIMENT=%q\n", cfg.CleanGOEXPERIMENT)
		}

		// TODO(rsc): Convince compiler team not to add more magic environment variables,
		// or perhaps restrict the environment variables passed to subprocesses.
		// Because these are clumsy, undocumented special-case hacks
		// for debugging the compiler, they are not settable using 'go env -w',
		// and so here we use os.Getenv, not cfg.Getenv.
		magic := []string{
			"GOCLOBBERDEADHASH",
			"GOSSAFUNC",
			"GOSSADIR",
			"GOCOMPILEDEBUG",
		}
		for _, env := range magic {
			if x := os.Getenv(env); x != "" {
				fmt.Fprintf(h, "magic %s=%s\n", env, x)
			}
		}

	case "gccgo":
		id, _, err := b.gccToolID(BuildToolchain.compiler(), "go")
		if err != nil {
			base.Fatalf("%v", err)
		}
		fmt.Fprintf(h, "compile %s %q %q\n", id, forcedGccgoflags, p.Internal.Gccgoflags)
		fmt.Fprintf(h, "pkgpath %s\n", gccgoPkgpath(p))
		fmt.Fprintf(h, "ar %q\n", BuildToolchain.(gccgoToolchain).ar())
		if len(p.SFiles) > 0 {
			id, _, _ = b.gccToolID(BuildToolchain.compiler(), "assembler-with-cpp")
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
		p.EmbedFiles,
	)
	for _, file := range inputFiles {
		fmt.Fprintf(h, "file %s %s\n", file, b.fileHash(filepath.Join(p.Dir, file)))
	}
	for _, a1 := range a.Deps {
		p1 := a1.Package
		if p1 != nil {
			fmt.Fprintf(h, "import %s %s\n", p1.ImportPath, contentID(a1.buildID))
		}
		if a1.Mode == "preprocess PGO profile" {
			fmt.Fprintf(h, "pgofile %s\n", b.fileHash(a1.built))
		}
	}

	return h.Sum()
}

// needCgoHdr reports whether the actions triggered by this one
// expect to be able to access the cgo-generated header file.
func (b *Builder) needCgoHdr(a *Action) bool {
	// If this build triggers a header install, run cgo to get the header.
	if !b.IsCmdList && (a.Package.UsesCgo() || a.Package.UsesSwig()) && (cfg.BuildBuildmode == "c-archive" || cfg.BuildBuildmode == "c-shared") {
		for _, t1 := range a.triggers {
			if t1.Mode == "install header" {
				return true
			}
		}
		for _, t1 := range a.triggers {
			for _, t2 := range t1.triggers {
				if t2.Mode == "install header" {
					return true
				}
			}
		}
	}
	return false
}

// allowedVersion reports whether the version v is an allowed version of go
// (one that we can compile).
// v is known to be of the form "1.23".
func allowedVersion(v string) bool {
	// Special case: no requirement.
	if v == "" {
		return true
	}
	return gover.Compare(gover.Local(), v) >= 0
}

const (
	needBuild uint32 = 1 << iota
	needCgoHdr
	needVet
	needCompiledGoFiles
	needCovMetaFile
	needStale
)

// build is the action for building a single package.
// Note that any new influence on this logic must be reported in b.buildActionID above as well.
func (b *Builder) build(ctx context.Context, a *Action) (err error) {
	p := a.Package
	sh := b.Shell(a)

	bit := func(x uint32, b bool) uint32 {
		if b {
			return x
		}
		return 0
	}

	cachedBuild := false
	needCovMeta := p.Internal.Cover.GenMeta
	need := bit(needBuild, !b.IsCmdList && a.needBuild || b.NeedExport) |
		bit(needCgoHdr, b.needCgoHdr(a)) |
		bit(needVet, a.needVet) |
		bit(needCovMetaFile, needCovMeta) |
		bit(needCompiledGoFiles, b.NeedCompiledGoFiles)

	if !p.BinaryOnly {
		if b.useCache(a, b.buildActionID(a), p.Target, need&needBuild != 0) {
			// We found the main output in the cache.
			// If we don't need any other outputs, we can stop.
			// Otherwise, we need to write files to a.Objdir (needVet, needCgoHdr).
			// Remember that we might have them in cache
			// and check again after we create a.Objdir.
			cachedBuild = true
			a.output = []byte{} // start saving output in case we miss any cache results
			need &^= needBuild
			if b.NeedExport {
				p.Export = a.built
				p.BuildID = a.buildID
			}
			if need&needCompiledGoFiles != 0 {
				if err := b.loadCachedCompiledGoFiles(a); err == nil {
					need &^= needCompiledGoFiles
				}
			}
		}

		// Source files might be cached, even if the full action is not
		// (e.g., go list -compiled -find).
		if !cachedBuild && need&needCompiledGoFiles != 0 {
			if err := b.loadCachedCompiledGoFiles(a); err == nil {
				need &^= needCompiledGoFiles
			}
		}

		if need == 0 {
			return nil
		}
		defer b.flushOutput(a)
	}

	defer func() {
		if err != nil && b.IsCmdList && b.NeedError && p.Error == nil {
			p.Error = &load.PackageError{Err: err}
		}
	}()
	if cfg.BuildN {
		// In -n mode, print a banner between packages.
		// The banner is five lines so that when changes to
		// different sections of the bootstrap script have to
		// be merged, the banners give patch something
		// to use to find its context.
		sh.Printf("\n#\n# %s\n#\n\n", p.ImportPath)
	}

	if cfg.BuildV {
		sh.Printf("%s\n", p.ImportPath)
	}

	if p.Error != nil {
		// Don't try to build anything for packages with errors. There may be a
		// problem with the inputs that makes the package unsafe to build.
		return p.Error
	}

	if p.BinaryOnly {
		p.Stale = true
		p.StaleReason = "binary-only packages are no longer supported"
		if b.IsCmdList {
			return nil
		}
		return errors.New("binary-only packages are no longer supported")
	}

	if p.Module != nil && !allowedVersion(p.Module.GoVersion) {
		return errors.New("module requires Go " + p.Module.GoVersion + " or later")
	}

	if err := b.checkDirectives(a); err != nil {
		return err
	}

	if err := sh.Mkdir(a.Objdir); err != nil {
		return err
	}
	objdir := a.Objdir

	// Load cached cgo header, but only if we're skipping the main build (cachedBuild==true).
	if cachedBuild && need&needCgoHdr != 0 {
		if err := b.loadCachedCgoHdr(a); err == nil {
			need &^= needCgoHdr
		}
	}

	// Load cached coverage meta-data file fragment, but only if we're
	// skipping the main build (cachedBuild==true).
	if cachedBuild && need&needCovMetaFile != 0 {
		bact := a.Actor.(*buildActor)
		if err := b.loadCachedObjdirFile(a, cache.Default(), bact.covMetaFileName); err == nil {
			need &^= needCovMetaFile
		}
	}

	// Load cached vet config, but only if that's all we have left
	// (need == needVet, not testing just the one bit).
	// If we are going to do a full build anyway,
	// we're going to regenerate the files below anyway.
	if need == needVet {
		if err := b.loadCachedVet(a); err == nil {
			need &^= needVet
		}
	}
	if need == 0 {
		return nil
	}

	if err := AllowInstall(a); err != nil {
		return err
	}

	// make target directory
	dir, _ := filepath.Split(a.Target)
	if dir != "" {
		if err := sh.Mkdir(dir); err != nil {
			return err
		}
	}

	gofiles := str.StringList(p.GoFiles)
	cgofiles := str.StringList(p.CgoFiles)
	cfiles := str.StringList(p.CFiles)
	sfiles := str.StringList(p.SFiles)
	cxxfiles := str.StringList(p.CXXFiles)
	var objects, cgoObjects, pcCFLAGS, pcLDFLAGS []string

	if p.UsesCgo() || p.UsesSwig() {
		if pcCFLAGS, pcLDFLAGS, err = b.getPkgConfigFlags(a); err != nil {
			return
		}
	}

	// Compute overlays for .c/.cc/.h/etc. and if there are any overlays
	// put correct contents of all those files in the objdir, to ensure
	// the correct headers are included. nonGoOverlay is the overlay that
	// points from nongo files to the copied files in objdir.
	nonGoFileLists := [][]string{p.CFiles, p.SFiles, p.CXXFiles, p.HFiles, p.FFiles}
OverlayLoop:
	for _, fs := range nonGoFileLists {
		for _, f := range fs {
			if fsys.Replaced(mkAbs(p.Dir, f)) {
				a.nonGoOverlay = make(map[string]string)
				break OverlayLoop
			}
		}
	}
	if a.nonGoOverlay != nil {
		for _, fs := range nonGoFileLists {
			for i := range fs {
				from := mkAbs(p.Dir, fs[i])
				dst := objdir + filepath.Base(fs[i])
				if err := sh.CopyFile(dst, fsys.Actual(from), 0666, false); err != nil {
					return err
				}
				a.nonGoOverlay[from] = dst
			}
		}
	}

	// If we're doing coverage, preprocess the .go files and put them in the work directory
	if p.Internal.Cover.Mode != "" {
		outfiles := []string{}
		infiles := []string{}
		for i, file := range str.StringList(gofiles, cgofiles) {
			if base.IsTestFile(file) {
				continue // Not covering this file.
			}

			var sourceFile string
			var coverFile string
			if base, found := strings.CutSuffix(file, ".cgo1.go"); found {
				// cgo files have absolute paths
				base = filepath.Base(base)
				sourceFile = file
				coverFile = objdir + base + ".cgo1.go"
			} else {
				sourceFile = filepath.Join(p.Dir, file)
				coverFile = objdir + file
			}
			coverFile = strings.TrimSuffix(coverFile, ".go") + ".cover.go"
			infiles = append(infiles, sourceFile)
			outfiles = append(outfiles, coverFile)
			if i < len(gofiles) {
				gofiles[i] = coverFile
			} else {
				cgofiles[i-len(gofiles)] = coverFile
			}
		}

		if len(infiles) != 0 {
			// Coverage instrumentation creates new top level
			// variables in the target package for things like
			// meta-data containers, counter vars, etc. To avoid
			// collisions with user variables, suffix the var name
			// with 12 hex digits from the SHA-256 hash of the
			// import path. Choice of 12 digits is historical/arbitrary,
			// we just need enough of the hash to avoid accidents,
			// as opposed to precluding determined attempts by
			// users to break things.
			sum := sha256.Sum256([]byte(a.Package.ImportPath))
			coverVar := fmt.Sprintf("goCover_%x_", sum[:6])
			mode := a.Package.Internal.Cover.Mode
			if mode == "" {
				panic("covermode should be set at this point")
			}
			if newoutfiles, err := b.cover(a, infiles, outfiles, coverVar, mode); err != nil {
				return err
			} else {
				outfiles = newoutfiles
				gofiles = append([]string{newoutfiles[0]}, gofiles...)
			}
			if ba, ok := a.Actor.(*buildActor); ok && ba.covMetaFileName != "" {
				b.cacheObjdirFile(a, cache.Default(), ba.covMetaFileName)
			}
		}
	}

	// Run SWIG on each .swig and .swigcxx file.
	// Each run will generate two files, a .go file and a .c or .cxx file.
	// The .go file will use import "C" and is to be processed by cgo.
	// For -cover test or build runs, this needs to happen after the cover
	// tool is run; we don't want to instrument swig-generated Go files,
	// see issue #64661.
	if p.UsesSwig() {
		outGo, outC, outCXX, err := b.swig(a, objdir, pcCFLAGS)
		if err != nil {
			return err
		}
		cgofiles = append(cgofiles, outGo...)
		cfiles = append(cfiles, outC...)
		cxxfiles = append(cxxfiles, outCXX...)
	}

	// Run cgo.
	if p.UsesCgo() || p.UsesSwig() {
		// In a package using cgo, cgo compiles the C, C++ and assembly files with gcc.
		// There is one exception: runtime/cgo's job is to bridge the
		// cgo and non-cgo worlds, so it necessarily has files in both.
		// In that case gcc only gets the gcc_* files.
		var gccfiles []string
		gccfiles = append(gccfiles, cfiles...)
		cfiles = nil
		if p.Standard && p.ImportPath == "runtime/cgo" {
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
				data, err := os.ReadFile(filepath.Join(p.Dir, sfile))
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

		outGo, outObj, err := b.cgo(a, base.Tool("cgo"), objdir, pcCFLAGS, pcLDFLAGS, mkAbsFiles(p.Dir, cgofiles), gccfiles, cxxfiles, p.MFiles, p.FFiles)

		// The files in cxxfiles have now been handled by b.cgo.
		cxxfiles = nil

		if err != nil {
			return err
		}
		if cfg.BuildToolchainName == "gccgo" {
			cgoObjects = append(cgoObjects, a.Objdir+"_cgo_flags")
		}
		cgoObjects = append(cgoObjects, outObj...)
		gofiles = append(gofiles, outGo...)

		switch cfg.BuildBuildmode {
		case "c-archive", "c-shared":
			b.cacheCgoHdr(a)
		}
	}

	var srcfiles []string // .go and non-.go
	srcfiles = append(srcfiles, gofiles...)
	srcfiles = append(srcfiles, sfiles...)
	srcfiles = append(srcfiles, cfiles...)
	srcfiles = append(srcfiles, cxxfiles...)
	b.cacheSrcFiles(a, srcfiles)

	// Running cgo generated the cgo header.
	need &^= needCgoHdr

	// Sanity check only, since Package.load already checked as well.
	if len(gofiles) == 0 {
		return &load.NoGoError{Package: p}
	}

	// Prepare Go vet config if needed.
	if need&needVet != 0 {
		buildVetConfig(a, srcfiles)
		need &^= needVet
	}
	if need&needCompiledGoFiles != 0 {
		if err := b.loadCachedCompiledGoFiles(a); err != nil {
			return fmt.Errorf("loading compiled Go files from cache: %w", err)
		}
		need &^= needCompiledGoFiles
	}
	if need == 0 {
		// Nothing left to do.
		return nil
	}

	// Collect symbol ABI requirements from assembly.
	symabis, err := BuildToolchain.symabis(b, a, sfiles)
	if err != nil {
		return err
	}

	// Prepare Go import config.
	// We start it off with a comment so it can't be empty, so icfg.Bytes() below is never nil.
	// It should never be empty anyway, but there have been bugs in the past that resulted
	// in empty configs, which then unfortunately turn into "no config passed to compiler",
	// and the compiler falls back to looking in pkg itself, which mostly works,
	// except when it doesn't.
	var icfg bytes.Buffer
	fmt.Fprintf(&icfg, "# import config\n")
	for i, raw := range p.Internal.RawImports {
		final := p.Imports[i]
		if final != raw {
			fmt.Fprintf(&icfg, "importmap %s=%s\n", raw, final)
		}
	}
	for _, a1 := range a.Deps {
		p1 := a1.Package
		if p1 == nil || p1.ImportPath == "" || a1.built == "" {
			continue
		}
		fmt.Fprintf(&icfg, "packagefile %s=%s\n", p1.ImportPath, a1.built)
	}

	// Prepare Go embed config if needed.
	// Unlike the import config, it's okay for the embed config to be empty.
	var embedcfg []byte
	if len(p.Internal.Embed) > 0 {
		var embed struct {
			Patterns map[string][]string
			Files    map[string]string
		}
		embed.Patterns = p.Internal.Embed
		embed.Files = make(map[string]string)
		for _, file := range p.EmbedFiles {
			embed.Files[file] = fsys.Actual(filepath.Join(p.Dir, file))
		}
		js, err := json.MarshalIndent(&embed, "", "\t")
		if err != nil {
			return fmt.Errorf("marshal embedcfg: %v", err)
		}
		embedcfg = js
	}

	// Find PGO profile if needed.
	var pgoProfile string
	for _, a1 := range a.Deps {
		if a1.Mode != "preprocess PGO profile" {
			continue
		}
		if pgoProfile != "" {
			return fmt.Errorf("action contains multiple PGO profile dependencies")
		}
		pgoProfile = a1.built
	}

	if p.Internal.BuildInfo != nil && cfg.ModulesEnabled {
		prog := modload.ModInfoProg(p.Internal.BuildInfo.String(), cfg.BuildToolchainName == "gccgo")
		if len(prog) > 0 {
			if err := sh.writeFile(objdir+"_gomod_.go", prog); err != nil {
				return err
			}
			gofiles = append(gofiles, objdir+"_gomod_.go")
		}
	}

	// Compile Go.
	objpkg := objdir + "_pkg_.a"
	ofile, out, err := BuildToolchain.gc(b, a, objpkg, icfg.Bytes(), embedcfg, symabis, len(sfiles) > 0, pgoProfile, gofiles)
	if err := sh.reportCmd("", "", out, err); err != nil {
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
	for _, file := range p.HFiles {
		name, ext := fileExtSplit(file)
		switch {
		case strings.HasSuffix(name, _goos_goarch):
			targ := file[:len(name)-len(_goos_goarch)] + "_GOOS_GOARCH." + ext
			if err := sh.CopyFile(objdir+targ, filepath.Join(p.Dir, file), 0666, true); err != nil {
				return err
			}
		case strings.HasSuffix(name, _goarch):
			targ := file[:len(name)-len(_goarch)] + "_GOARCH." + ext
			if err := sh.CopyFile(objdir+targ, filepath.Join(p.Dir, file), 0666, true); err != nil {
				return err
			}
		case strings.HasSuffix(name, _goos):
			targ := file[:len(name)-len(_goos)] + "_GOOS." + ext
			if err := sh.CopyFile(objdir+targ, filepath.Join(p.Dir, file), 0666, true); err != nil {
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
	// This lets us set the SHF_EXCLUDE flag.
	// This is read by readGccgoArchive in cmd/internal/buildid/buildid.go.
	if a.buildID != "" && cfg.BuildToolchainName == "gccgo" {
		switch cfg.Goos {
		case "aix", "android", "dragonfly", "freebsd", "illumos", "linux", "netbsd", "openbsd", "solaris":
			asmfile, err := b.gccgoBuildIDFile(a)
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
	for _, syso := range p.SysoFiles {
		objects = append(objects, filepath.Join(p.Dir, syso))
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

	if err := b.updateBuildID(a, objpkg); err != nil {
		return err
	}

	a.built = objpkg
	return nil
}

func (b *Builder) checkDirectives(a *Action) error {
	var msg []byte
	p := a.Package
	var seen map[string]token.Position
	for _, d := range p.Internal.Build.Directives {
		if strings.HasPrefix(d.Text, "//go:debug") {
			key, _, err := load.ParseGoDebug(d.Text)
			if err != nil && err != load.ErrNotGoDebug {
				msg = fmt.Appendf(msg, "%s: invalid //go:debug: %v\n", d.Pos, err)
				continue
			}
			if pos, ok := seen[key]; ok {
				msg = fmt.Appendf(msg, "%s: repeated //go:debug for %v\n\t%s: previous //go:debug\n", d.Pos, key, pos)
				continue
			}
			if seen == nil {
				seen = make(map[string]token.Position)
			}
			seen[key] = d.Pos
		}
	}
	if len(msg) > 0 {
		// We pass a non-nil error to reportCmd to trigger the failure reporting
		// path, but the content of the error doesn't matter because msg is
		// non-empty.
		err := errors.New("invalid directive")
		return b.Shell(a).reportCmd("", "", msg, err)
	}
	return nil
}

func (b *Builder) cacheObjdirFile(a *Action, c cache.Cache, name string) error {
	f, err := os.Open(a.Objdir + name)
	if err != nil {
		return err
	}
	defer f.Close()
	_, _, err = c.Put(cache.Subkey(a.actionID, name), f)
	return err
}

func (b *Builder) findCachedObjdirFile(a *Action, c cache.Cache, name string) (string, error) {
	file, _, err := cache.GetFile(c, cache.Subkey(a.actionID, name))
	if err != nil {
		return "", fmt.Errorf("loading cached file %s: %w", name, err)
	}
	return file, nil
}

func (b *Builder) loadCachedObjdirFile(a *Action, c cache.Cache, name string) error {
	cached, err := b.findCachedObjdirFile(a, c, name)
	if err != nil {
		return err
	}
	return b.Shell(a).CopyFile(a.Objdir+name, cached, 0666, true)
}

func (b *Builder) cacheCgoHdr(a *Action) {
	c := cache.Default()
	b.cacheObjdirFile(a, c, "_cgo_install.h")
}

func (b *Builder) loadCachedCgoHdr(a *Action) error {
	c := cache.Default()
	return b.loadCachedObjdirFile(a, c, "_cgo_install.h")
}

func (b *Builder) cacheSrcFiles(a *Action, srcfiles []string) {
	c := cache.Default()
	var buf bytes.Buffer
	for _, file := range srcfiles {
		if !strings.HasPrefix(file, a.Objdir) {
			// not generated
			buf.WriteString("./")
			buf.WriteString(file)
			buf.WriteString("\n")
			continue
		}
		name := file[len(a.Objdir):]
		buf.WriteString(name)
		buf.WriteString("\n")
		if err := b.cacheObjdirFile(a, c, name); err != nil {
			return
		}
	}
	cache.PutBytes(c, cache.Subkey(a.actionID, "srcfiles"), buf.Bytes())
}

func (b *Builder) loadCachedVet(a *Action) error {
	c := cache.Default()
	list, _, err := cache.GetBytes(c, cache.Subkey(a.actionID, "srcfiles"))
	if err != nil {
		return fmt.Errorf("reading srcfiles list: %w", err)
	}
	var srcfiles []string
	for _, name := range strings.Split(string(list), "\n") {
		if name == "" { // end of list
			continue
		}
		if strings.HasPrefix(name, "./") {
			srcfiles = append(srcfiles, name[2:])
			continue
		}
		if err := b.loadCachedObjdirFile(a, c, name); err != nil {
			return err
		}
		srcfiles = append(srcfiles, a.Objdir+name)
	}
	buildVetConfig(a, srcfiles)
	return nil
}

func (b *Builder) loadCachedCompiledGoFiles(a *Action) error {
	c := cache.Default()
	list, _, err := cache.GetBytes(c, cache.Subkey(a.actionID, "srcfiles"))
	if err != nil {
		return fmt.Errorf("reading srcfiles list: %w", err)
	}
	var gofiles []string
	for _, name := range strings.Split(string(list), "\n") {
		if name == "" { // end of list
			continue
		} else if !strings.HasSuffix(name, ".go") {
			continue
		}
		if strings.HasPrefix(name, "./") {
			gofiles = append(gofiles, name[len("./"):])
			continue
		}
		file, err := b.findCachedObjdirFile(a, c, name)
		if err != nil {
			return fmt.Errorf("finding %s: %w", name, err)
		}
		gofiles = append(gofiles, file)
	}
	a.Package.CompiledGoFiles = gofiles
	return nil
}

// vetConfig is the configuration passed to vet describing a single package.
type vetConfig struct {
	ID           string   // package ID (example: "fmt [fmt.test]")
	Compiler     string   // compiler name (gc, gccgo)
	Dir          string   // directory containing package
	ImportPath   string   // canonical import path ("package path")
	GoFiles      []string // absolute paths to package source files
	NonGoFiles   []string // absolute paths to package non-Go files
	IgnoredFiles []string // absolute paths to ignored source files

	ModulePath    string            // module path (may be "" on module error)
	ModuleVersion string            // module version (may be "" on main module or module error)
	ImportMap     map[string]string // map import path in source code to package path
	PackageFile   map[string]string // map package path to .a file with export data
	Standard      map[string]bool   // map package path to whether it's in the standard library
	PackageVetx   map[string]string // map package path to vetx data from earlier vet run
	VetxOnly      bool              // only compute vetx data; don't report detected problems
	VetxOutput    string            // write vetx data to this output file
	GoVersion     string            // Go version for package

	SucceedOnTypecheckFailure bool // awful hack; see #18395 and below
}

func buildVetConfig(a *Action, srcfiles []string) {
	// Classify files based on .go extension.
	// srcfiles does not include raw cgo files.
	var gofiles, nongofiles []string
	for _, name := range srcfiles {
		if strings.HasSuffix(name, ".go") {
			gofiles = append(gofiles, name)
		} else {
			nongofiles = append(nongofiles, name)
		}
	}

	ignored := str.StringList(a.Package.IgnoredGoFiles, a.Package.IgnoredOtherFiles)

	// Pass list of absolute paths to vet,
	// so that vet's error messages will use absolute paths,
	// so that we can reformat them relative to the directory
	// in which the go command is invoked.
	vcfg := &vetConfig{
		ID:           a.Package.ImportPath,
		Compiler:     cfg.BuildToolchainName,
		Dir:          a.Package.Dir,
		GoFiles:      actualFiles(mkAbsFiles(a.Package.Dir, gofiles)),
		NonGoFiles:   actualFiles(mkAbsFiles(a.Package.Dir, nongofiles)),
		IgnoredFiles: actualFiles(mkAbsFiles(a.Package.Dir, ignored)),
		ImportPath:   a.Package.ImportPath,
		ImportMap:    make(map[string]string),
		PackageFile:  make(map[string]string),
		Standard:     make(map[string]bool),
	}
	vcfg.GoVersion = "go" + gover.Local()
	if a.Package.Module != nil {
		v := a.Package.Module.GoVersion
		if v == "" {
			v = gover.DefaultGoModVersion
		}
		vcfg.GoVersion = "go" + v

		if a.Package.Module.Error == nil {
			vcfg.ModulePath = a.Package.Module.Path
			vcfg.ModuleVersion = a.Package.Module.Version
		}
	}
	a.vetCfg = vcfg
	for i, raw := range a.Package.Internal.RawImports {
		final := a.Package.Imports[i]
		vcfg.ImportMap[raw] = final
	}

	// Compute the list of mapped imports in the vet config
	// so that we can add any missing mappings below.
	vcfgMapped := make(map[string]bool)
	for _, p := range vcfg.ImportMap {
		vcfgMapped[p] = true
	}

	for _, a1 := range a.Deps {
		p1 := a1.Package
		if p1 == nil || p1.ImportPath == "" {
			continue
		}
		// Add import mapping if needed
		// (for imports like "runtime/cgo" that appear only in generated code).
		if !vcfgMapped[p1.ImportPath] {
			vcfg.ImportMap[p1.ImportPath] = p1.ImportPath
		}
		if a1.built != "" {
			vcfg.PackageFile[p1.ImportPath] = a1.built
		}
		if p1.Standard {
			vcfg.Standard[p1.ImportPath] = true
		}
	}
}

// VetTool is the path to an alternate vet tool binary.
// The caller is expected to set it (if needed) before executing any vet actions.
var VetTool string

// VetFlags are the default flags to pass to vet.
// The caller is expected to set them before executing any vet actions.
var VetFlags []string

// VetExplicit records whether the vet flags were set explicitly on the command line.
var VetExplicit bool

func (b *Builder) vet(ctx context.Context, a *Action) error {
	// a.Deps[0] is the build of the package being vetted.
	// a.Deps[1] is the build of the "fmt" package.

	a.Failed = nil // vet of dependency may have failed but we can still succeed

	if a.Deps[0].Failed != nil {
		// The build of the package has failed. Skip vet check.
		// Vet could return export data for non-typecheck errors,
		// but we ignore it because the package cannot be compiled.
		return nil
	}

	vcfg := a.Deps[0].vetCfg
	if vcfg == nil {
		// Vet config should only be missing if the build failed.
		return fmt.Errorf("vet config not found")
	}

	sh := b.Shell(a)

	vcfg.VetxOnly = a.VetxOnly
	vcfg.VetxOutput = a.Objdir + "vet.out"
	vcfg.PackageVetx = make(map[string]string)

	h := cache.NewHash("vet " + a.Package.ImportPath)
	fmt.Fprintf(h, "vet %q\n", b.toolID("vet"))

	vetFlags := VetFlags

	// In GOROOT, we enable all the vet tests during 'go test',
	// not just the high-confidence subset. This gets us extra
	// checking for the standard library (at some compliance cost)
	// and helps us gain experience about how well the checks
	// work, to help decide which should be turned on by default.
	// The command-line still wins.
	//
	// Note that this flag change applies even when running vet as
	// a dependency of vetting a package outside std.
	// (Otherwise we'd have to introduce a whole separate
	// space of "vet fmt as a dependency of a std top-level vet"
	// versus "vet fmt as a dependency of a non-std top-level vet".)
	// This is OK as long as the packages that are farther down the
	// dependency tree turn on *more* analysis, as here.
	// (The unsafeptr check does not write any facts for use by
	// later vet runs, nor does unreachable.)
	if a.Package.Goroot && !VetExplicit && VetTool == "" {
		// Turn off -unsafeptr checks.
		// There's too much unsafe.Pointer code
		// that vet doesn't like in low-level packages
		// like runtime, sync, and reflect.
		// Note that $GOROOT/src/buildall.bash
		// does the same
		// and should be updated if these flags are
		// changed here.
		vetFlags = []string{"-unsafeptr=false"}

		// Also turn off -unreachable checks during go test.
		// During testing it is very common to make changes
		// like hard-coded forced returns or panics that make
		// code unreachable. It's unreasonable to insist on files
		// not having any unreachable code during "go test".
		// (buildall.bash still has -unreachable enabled
		// for the overall whole-tree scan.)
		if cfg.CmdName == "test" {
			vetFlags = append(vetFlags, "-unreachable=false")
		}
	}

	// Note: We could decide that vet should compute export data for
	// all analyses, in which case we don't need to include the flags here.
	// But that would mean that if an analysis causes problems like
	// unexpected crashes there would be no way to turn it off.
	// It seems better to let the flags disable export analysis too.
	fmt.Fprintf(h, "vetflags %q\n", vetFlags)

	fmt.Fprintf(h, "pkg %q\n", a.Deps[0].actionID)
	for _, a1 := range a.Deps {
		if a1.Mode == "vet" && a1.built != "" {
			fmt.Fprintf(h, "vetout %q %s\n", a1.Package.ImportPath, b.fileHash(a1.built))
			vcfg.PackageVetx[a1.Package.ImportPath] = a1.built
		}
	}
	key := cache.ActionID(h.Sum())

	if vcfg.VetxOnly && !cfg.BuildA {
		c := cache.Default()
		if file, _, err := cache.GetFile(c, key); err == nil {
			a.built = file
			return nil
		}
	}

	js, err := json.MarshalIndent(vcfg, "", "\t")
	if err != nil {
		return fmt.Errorf("internal error marshaling vet config: %v", err)
	}
	js = append(js, '\n')
	if err := sh.writeFile(a.Objdir+"vet.cfg", js); err != nil {
		return err
	}

	// TODO(rsc): Why do we pass $GCCGO to go vet?
	env := b.cCompilerEnv()
	if cfg.BuildToolchainName == "gccgo" {
		env = append(env, "GCCGO="+BuildToolchain.compiler())
	}

	p := a.Package
	tool := VetTool
	if tool == "" {
		tool = base.Tool("vet")
	}
	runErr := sh.run(p.Dir, p.ImportPath, env, cfg.BuildToolexec, tool, vetFlags, a.Objdir+"vet.cfg")

	// If vet wrote export data, save it for input to future vets.
	if f, err := os.Open(vcfg.VetxOutput); err == nil {
		a.built = vcfg.VetxOutput
		cache.Default().Put(key, f)
		f.Close()
	}

	return runErr
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
	fmt.Fprintf(h, "defaultgodebug %q\n", p.DefaultGODEBUG)
	if cfg.BuildTrimpath {
		fmt.Fprintln(h, "trimpath")
	}

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

		// GOARM, GOMIPS, etc.
		key, val, _ := cfg.GetArchEnv()
		fmt.Fprintf(h, "%s=%s\n", key, val)

		if cfg.CleanGOEXPERIMENT != "" {
			fmt.Fprintf(h, "GOEXPERIMENT=%q\n", cfg.CleanGOEXPERIMENT)
		}

		// The linker writes source file paths that refer to GOROOT,
		// but only if -trimpath is not specified (see [gctoolchain.ld] in gc.go).
		gorootFinal := cfg.GOROOT
		if cfg.BuildTrimpath {
			gorootFinal = ""
		}
		fmt.Fprintf(h, "GOROOT=%s\n", gorootFinal)

		// GO_EXTLINK_ENABLED controls whether the external linker is used.
		fmt.Fprintf(h, "GO_EXTLINK_ENABLED=%s\n", cfg.Getenv("GO_EXTLINK_ENABLED"))

		// TODO(rsc): Do cgo settings and flags need to be included?
		// Or external linker settings and flags?

	case "gccgo":
		id, _, err := b.gccToolID(BuildToolchain.linker(), "go")
		if err != nil {
			base.Fatalf("%v", err)
		}
		fmt.Fprintf(h, "link %s %s\n", id, ldBuildmode)
		// TODO(iant): Should probably include cgo flags here.
	}
}

// link is the action for linking a single command.
// Note that any new influence on this logic must be reported in b.linkActionID above as well.
func (b *Builder) link(ctx context.Context, a *Action) (err error) {
	if b.useCache(a, b.linkActionID(a), a.Package.Target, !b.IsCmdList) || b.IsCmdList {
		return nil
	}
	defer b.flushOutput(a)

	sh := b.Shell(a)
	if err := sh.Mkdir(a.Objdir); err != nil {
		return err
	}

	importcfg := a.Objdir + "importcfg.link"
	if err := b.writeLinkImportcfg(a, importcfg); err != nil {
		return err
	}

	if err := AllowInstall(a); err != nil {
		return err
	}

	// make target directory
	dir, _ := filepath.Split(a.Target)
	if dir != "" {
		if err := sh.Mkdir(dir); err != nil {
			return err
		}
	}

	if err := BuildToolchain.ld(b, a, a.Target, importcfg, a.Deps[0].built); err != nil {
		return err
	}

	// Update the binary with the final build ID.
	if err := b.updateBuildID(a, a.Target); err != nil {
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
	info := ""
	if a.Package.Internal.BuildInfo != nil {
		info = a.Package.Internal.BuildInfo.String()
	}
	fmt.Fprintf(&icfg, "modinfo %q\n", modload.ModInfoData(info))
	return b.Shell(a).writeFile(file, icfg.Bytes())
}

// PkgconfigCmd returns a pkg-config binary name
// defaultPkgConfig is defined in zdefaultcc.go, written by cmd/dist.
func (b *Builder) PkgconfigCmd() string {
	return envList("PKG_CONFIG", cfg.DefaultPkgConfig)[0]
}

// splitPkgConfigOutput parses the pkg-config output into a slice of flags.
// This implements the shell quoting semantics described in
// https://pubs.opengroup.org/onlinepubs/9699919799/utilities/V3_chap02.html#tag_18_02,
// except that it does not support parameter or arithmetic expansion or command
// substitution and hard-codes the <blank> delimiters instead of reading them
// from LC_LOCALE.
func splitPkgConfigOutput(out []byte) ([]string, error) {
	if len(out) == 0 {
		return nil, nil
	}
	var flags []string
	flag := make([]byte, 0, len(out))
	didQuote := false // was the current flag parsed from a quoted string?
	escaped := false  // did we just read `\` in a non-single-quoted context?
	quote := byte(0)  // what is the quote character around the current string?

	for _, c := range out {
		if escaped {
			if quote == '"' {
				// “The <backslash> shall retain its special meaning as an escape
				// character … only when followed by one of the following characters
				// when considered special:”
				switch c {
				case '$', '`', '"', '\\', '\n':
					// Handle the escaped character normally.
				default:
					// Not an escape character after all.
					flag = append(flag, '\\', c)
					escaped = false
					continue
				}
			}

			if c == '\n' {
				// “If a <newline> follows the <backslash>, the shell shall interpret
				// this as line continuation.”
			} else {
				flag = append(flag, c)
			}
			escaped = false
			continue
		}

		if quote != 0 && c == quote {
			quote = 0
			continue
		}
		switch quote {
		case '\'':
			// “preserve the literal value of each character”
			flag = append(flag, c)
			continue
		case '"':
			// “preserve the literal value of all characters within the double-quotes,
			// with the exception of …”
			switch c {
			case '`', '$', '\\':
			default:
				flag = append(flag, c)
				continue
			}
		}

		// “The application shall quote the following characters if they are to
		// represent themselves:”
		switch c {
		case '|', '&', ';', '<', '>', '(', ')', '$', '`':
			return nil, fmt.Errorf("unexpected shell character %q in pkgconf output", c)

		case '\\':
			// “A <backslash> that is not quoted shall preserve the literal value of
			// the following character, with the exception of a <newline>.”
			escaped = true
			continue

		case '"', '\'':
			quote = c
			didQuote = true
			continue

		case ' ', '\t', '\n':
			if len(flag) > 0 || didQuote {
				flags = append(flags, string(flag))
			}
			flag, didQuote = flag[:0], false
			continue
		}

		flag = append(flag, c)
	}

	// Prefer to report a missing quote instead of a missing escape. If the string
	// is something like `"foo\`, it's ambiguous as to whether the trailing
	// backslash is really an escape at all.
	if quote != 0 {
		return nil, errors.New("unterminated quoted string in pkgconf output")
	}
	if escaped {
		return nil, errors.New("broken character escaping in pkgconf output")
	}

	if len(flag) > 0 || didQuote {
		flags = append(flags, string(flag))
	}
	return flags, nil
}

// Calls pkg-config if needed and returns the cflags/ldflags needed to build a's package.
func (b *Builder) getPkgConfigFlags(a *Action) (cflags, ldflags []string, err error) {
	p := a.Package
	sh := b.Shell(a)
	if pcargs := p.CgoPkgConfig; len(pcargs) > 0 {
		// pkg-config permits arguments to appear anywhere in
		// the command line. Move them all to the front, before --.
		var pcflags []string
		var pkgs []string
		for _, pcarg := range pcargs {
			if pcarg == "--" {
				// We're going to add our own "--" argument.
			} else if strings.HasPrefix(pcarg, "--") {
				pcflags = append(pcflags, pcarg)
			} else {
				pkgs = append(pkgs, pcarg)
			}
		}
		for _, pkg := range pkgs {
			if !load.SafeArg(pkg) {
				return nil, nil, fmt.Errorf("invalid pkg-config package name: %s", pkg)
			}
		}
		var out []byte
		out, err = sh.runOut(p.Dir, nil, b.PkgconfigCmd(), "--cflags", pcflags, "--", pkgs)
		if err != nil {
			desc := b.PkgconfigCmd() + " --cflags " + strings.Join(pcflags, " ") + " -- " + strings.Join(pkgs, " ")
			return nil, nil, sh.reportCmd(desc, "", out, err)
		}
		if len(out) > 0 {
			cflags, err = splitPkgConfigOutput(bytes.TrimSpace(out))
			if err != nil {
				return nil, nil, err
			}
			if err := checkCompilerFlags("CFLAGS", "pkg-config --cflags", cflags); err != nil {
				return nil, nil, err
			}
		}
		out, err = sh.runOut(p.Dir, nil, b.PkgconfigCmd(), "--libs", pcflags, "--", pkgs)
		if err != nil {
			desc := b.PkgconfigCmd() + " --libs " + strings.Join(pcflags, " ") + " -- " + strings.Join(pkgs, " ")
			return nil, nil, sh.reportCmd(desc, "", out, err)
		}
		if len(out) > 0 {
			// We need to handle path with spaces so that C:/Program\ Files can pass
			// checkLinkerFlags. Use splitPkgConfigOutput here just like we treat cflags.
			ldflags, err = splitPkgConfigOutput(bytes.TrimSpace(out))
			if err != nil {
				return nil, nil, err
			}
			if err := checkLinkerFlags("LDFLAGS", "pkg-config --libs", ldflags); err != nil {
				return nil, nil, err
			}
		}
	}

	return
}

func (b *Builder) installShlibname(ctx context.Context, a *Action) error {
	if err := AllowInstall(a); err != nil {
		return err
	}

	sh := b.Shell(a)
	a1 := a.Deps[0]
	if !cfg.BuildN {
		if err := sh.Mkdir(filepath.Dir(a.Target)); err != nil {
			return err
		}
	}
	return sh.writeFile(a.Target, []byte(filepath.Base(a1.Target)+"\n"))
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

func (b *Builder) linkShared(ctx context.Context, a *Action) (err error) {
	if b.useCache(a, b.linkSharedActionID(a), a.Target, !b.IsCmdList) || b.IsCmdList {
		return nil
	}
	defer b.flushOutput(a)

	if err := AllowInstall(a); err != nil {
		return err
	}

	if err := b.Shell(a).Mkdir(a.Objdir); err != nil {
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
func BuildInstallFunc(b *Builder, ctx context.Context, a *Action) (err error) {
	defer func() {
		if err != nil {
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
	sh := b.Shell(a)

	a1 := a.Deps[0]
	a.buildID = a1.buildID
	if a.json != nil {
		a.json.BuildID = a.buildID
	}

	// If we are using the eventual install target as an up-to-date
	// cached copy of the thing we built, then there's no need to
	// copy it into itself (and that would probably fail anyway).
	// In this case a1.built == a.Target because a1.built == p.Target,
	// so the built target is not in the a1.Objdir tree that b.cleanup(a1) removes.
	if a1.built == a.Target {
		a.built = a.Target
		if !a.buggyInstall {
			b.cleanup(a1)
		}
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
		if !a.buggyInstall && !b.IsCmdList {
			if cfg.BuildN {
				sh.ShowCmd("", "touch %s", a.Target)
			} else if err := AllowInstall(a); err == nil {
				now := time.Now()
				os.Chtimes(a.Target, now, now)
			}
		}
		return nil
	}

	// If we're building for go list -export,
	// never install anything; just keep the cache reference.
	if b.IsCmdList {
		a.built = a1.built
		return nil
	}
	if err := AllowInstall(a); err != nil {
		return err
	}

	if err := sh.Mkdir(a.Objdir); err != nil {
		return err
	}

	perm := fs.FileMode(0666)
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
		if err := sh.Mkdir(dir); err != nil {
			return err
		}
	}

	if !a.buggyInstall {
		defer b.cleanup(a1)
	}

	return sh.moveOrCopyFile(a.Target, a1.built, perm, false)
}

// AllowInstall returns a non-nil error if this invocation of the go command is
// allowed to install a.Target.
//
// The build of cmd/go running under its own test is forbidden from installing
// to its original GOROOT. The var is exported so it can be set by TestMain.
var AllowInstall = func(*Action) error { return nil }

// cleanup removes a's object dir to keep the amount of
// on-disk garbage down in a large build. On an operating system
// with aggressive buffering, cleaning incrementally like
// this keeps the intermediate objects from hitting the disk.
func (b *Builder) cleanup(a *Action) {
	if !cfg.BuildWork {
		b.Shell(a).RemoveAll(a.Objdir)
	}
}

// Install the cgo export header file, if there is one.
func (b *Builder) installHeader(ctx context.Context, a *Action) error {
	sh := b.Shell(a)

	src := a.Objdir + "_cgo_install.h"
	if _, err := os.Stat(src); os.IsNotExist(err) {
		// If the file does not exist, there are no exported
		// functions, and we do not install anything.
		// TODO(rsc): Once we know that caching is rebuilding
		// at the right times (not missing rebuilds), here we should
		// probably delete the installed header, if any.
		if cfg.BuildX {
			sh.ShowCmd("", "# %s not created", src)
		}
		return nil
	}

	if err := AllowInstall(a); err != nil {
		return err
	}

	dir, _ := filepath.Split(a.Target)
	if dir != "" {
		if err := sh.Mkdir(dir); err != nil {
			return err
		}
	}

	return sh.moveOrCopyFile(a.Target, src, 0666, true)
}

// cover runs, in effect,
//
//	go tool cover -pkgcfg=<config file> -mode=b.coverMode -var="varName" -o <outfiles> <infiles>
//
// Return value is an updated output files list; in addition to the
// regular outputs (instrumented source files) the cover tool also
// writes a separate file (appearing first in the list of outputs)
// that will contain coverage counters and meta-data.
func (b *Builder) cover(a *Action, infiles, outfiles []string, varName string, mode string) ([]string, error) {
	pkgcfg := a.Objdir + "pkgcfg.txt"
	covoutputs := a.Objdir + "coveroutfiles.txt"
	odir := filepath.Dir(outfiles[0])
	cv := filepath.Join(odir, "covervars.go")
	outfiles = append([]string{cv}, outfiles...)
	if err := b.writeCoverPkgInputs(a, pkgcfg, covoutputs, outfiles); err != nil {
		return nil, err
	}
	args := []string{base.Tool("cover"),
		"-pkgcfg", pkgcfg,
		"-mode", mode,
		"-var", varName,
		"-outfilelist", covoutputs,
	}
	args = append(args, infiles...)
	if err := b.Shell(a).run(a.Objdir, "", nil,
		cfg.BuildToolexec, args); err != nil {
		return nil, err
	}
	return outfiles, nil
}

func (b *Builder) writeCoverPkgInputs(a *Action, pconfigfile string, covoutputsfile string, outfiles []string) error {
	sh := b.Shell(a)
	p := a.Package
	p.Internal.Cover.Cfg = a.Objdir + "coveragecfg"
	pcfg := covcmd.CoverPkgConfig{
		PkgPath: p.ImportPath,
		PkgName: p.Name,
		// Note: coverage granularity is currently hard-wired to
		// 'perblock'; there isn't a way using "go build -cover" or "go
		// test -cover" to select it. This may change in the future
		// depending on user demand.
		Granularity: "perblock",
		OutConfig:   p.Internal.Cover.Cfg,
		Local:       p.Internal.Local,
	}
	if ba, ok := a.Actor.(*buildActor); ok && ba.covMetaFileName != "" {
		pcfg.EmitMetaFile = a.Objdir + ba.covMetaFileName
	}
	if a.Package.Module != nil {
		pcfg.ModulePath = a.Package.Module.Path
	}
	data, err := json.Marshal(pcfg)
	if err != nil {
		return err
	}
	data = append(data, '\n')
	if err := sh.writeFile(pconfigfile, data); err != nil {
		return err
	}
	var sb strings.Builder
	for i := range outfiles {
		fmt.Fprintf(&sb, "%s\n", outfiles[i])
	}
	return sh.writeFile(covoutputsfile, []byte(sb.String()))
}

var objectMagic = [][]byte{
	{'!', '<', 'a', 'r', 'c', 'h', '>', '\n'}, // Package archive
	{'<', 'b', 'i', 'g', 'a', 'f', '>', '\n'}, // Package AIX big archive
	{'\x7F', 'E', 'L', 'F'},                   // ELF
	{0xFE, 0xED, 0xFA, 0xCE},                  // Mach-O big-endian 32-bit
	{0xFE, 0xED, 0xFA, 0xCF},                  // Mach-O big-endian 64-bit
	{0xCE, 0xFA, 0xED, 0xFE},                  // Mach-O little-endian 32-bit
	{0xCF, 0xFA, 0xED, 0xFE},                  // Mach-O little-endian 64-bit
	{0x4d, 0x5a, 0x90, 0x00, 0x03, 0x00},      // PE (Windows) as generated by 6l/8l and gcc
	{0x4d, 0x5a, 0x78, 0x00, 0x01, 0x00},      // PE (Windows) as generated by llvm for dll
	{0x00, 0x00, 0x01, 0xEB},                  // Plan 9 i386
	{0x00, 0x00, 0x8a, 0x97},                  // Plan 9 amd64
	{0x00, 0x00, 0x06, 0x47},                  // Plan 9 arm
	{0x00, 0x61, 0x73, 0x6D},                  // WASM
	{0x01, 0xDF},                              // XCOFF 32bit
	{0x01, 0xF7},                              // XCOFF 64bit
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

// cCompilerEnv returns environment variables to set when running the
// C compiler. This is needed to disable escape codes in clang error
// messages that confuse tools like cgo.
func (b *Builder) cCompilerEnv() []string {
	return []string{"TERM=dumb"}
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
	gc(b *Builder, a *Action, archive string, importcfg, embedcfg []byte, symabis string, asmhdr bool, pgoProfile string, gofiles []string) (ofile string, out []byte, err error)
	// cc runs the toolchain's C compiler in a directory on a C file
	// to produce an output file.
	cc(b *Builder, a *Action, ofile, cfile string) error
	// asm runs the assembler in a specific directory on specific files
	// and returns a list of named output files.
	asm(b *Builder, a *Action, sfiles []string) ([]string, error)
	// symabis scans the symbol ABIs from sfiles and returns the
	// path to the output symbol ABIs file, or "" if none.
	symabis(b *Builder, a *Action, sfiles []string) (string, error)
	// pack runs the archive packer in a specific directory to create
	// an archive from a set of object files.
	// typically it is run in the object directory.
	pack(b *Builder, a *Action, afile string, ofiles []string) error
	// ld runs the linker to create an executable starting at mainpkg.
	ld(b *Builder, root *Action, targetPath, importcfg, mainpkg string) error
	// ldShared runs the linker to create a shared library containing the pkgs built by toplevelactions
	ldShared(b *Builder, root *Action, toplevelactions []*Action, targetPath, importcfg string, allactions []*Action) error

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

func (noToolchain) gc(b *Builder, a *Action, archive string, importcfg, embedcfg []byte, symabis string, asmhdr bool, pgoProfile string, gofiles []string) (ofile string, out []byte, err error) {
	return "", nil, noCompiler()
}

func (noToolchain) asm(b *Builder, a *Action, sfiles []string) ([]string, error) {
	return nil, noCompiler()
}

func (noToolchain) symabis(b *Builder, a *Action, sfiles []string) (string, error) {
	return "", noCompiler()
}

func (noToolchain) pack(b *Builder, a *Action, afile string, ofiles []string) error {
	return noCompiler()
}

func (noToolchain) ld(b *Builder, root *Action, targetPath, importcfg, mainpkg string) error {
	return noCompiler()
}

func (noToolchain) ldShared(b *Builder, root *Action, toplevelactions []*Action, targetPath, importcfg string, allactions []*Action) error {
	return noCompiler()
}

func (noToolchain) cc(b *Builder, a *Action, ofile, cfile string) error {
	return noCompiler()
}

// gcc runs the gcc C compiler to create an object from a single C file.
func (b *Builder) gcc(a *Action, workdir, out string, flags []string, cfile string) error {
	p := a.Package
	return b.ccompile(a, out, flags, cfile, b.GccCmd(p.Dir, workdir))
}

// gxx runs the g++ C++ compiler to create an object from a single C++ file.
func (b *Builder) gxx(a *Action, workdir, out string, flags []string, cxxfile string) error {
	p := a.Package
	return b.ccompile(a, out, flags, cxxfile, b.GxxCmd(p.Dir, workdir))
}

// gfortran runs the gfortran Fortran compiler to create an object from a single Fortran file.
func (b *Builder) gfortran(a *Action, workdir, out string, flags []string, ffile string) error {
	p := a.Package
	return b.ccompile(a, out, flags, ffile, b.gfortranCmd(p.Dir, workdir))
}

// ccompile runs the given C or C++ compiler and creates an object from a single source file.
func (b *Builder) ccompile(a *Action, outfile string, flags []string, file string, compiler []string) error {
	p := a.Package
	sh := b.Shell(a)
	file = mkAbs(p.Dir, file)
	outfile = mkAbs(p.Dir, outfile)

	// Elide source directory paths if -trimpath is set.
	// This is needed for source files (e.g., a .c file in a package directory).
	// TODO(golang.org/issue/36072): cgo also generates files with #line
	// directives pointing to the source directory. It should not generate those
	// when -trimpath is enabled.
	if b.gccSupportsFlag(compiler, "-fdebug-prefix-map=a=b") {
		if cfg.BuildTrimpath || p.Goroot {
			prefixMapFlag := "-fdebug-prefix-map"
			if b.gccSupportsFlag(compiler, "-ffile-prefix-map=a=b") {
				prefixMapFlag = "-ffile-prefix-map"
			}
			// Keep in sync with Action.trimpath.
			// The trimmed paths are a little different, but we need to trim in mostly the
			// same situations.
			var from, toPath string
			if m := p.Module; m == nil {
				if p.Root == "" { // command-line-arguments in GOPATH mode, maybe?
					from = p.Dir
					toPath = p.ImportPath
				} else if p.Goroot {
					from = p.Root
					toPath = "GOROOT"
				} else {
					from = p.Root
					toPath = "GOPATH"
				}
			} else if m.Dir == "" {
				// The module is in the vendor directory. Replace the entire vendor
				// directory path, because the module's Dir is not filled in.
				from = modload.VendorDir()
				toPath = "vendor"
			} else {
				from = m.Dir
				toPath = m.Path
				if m.Version != "" {
					toPath += "@" + m.Version
				}
			}
			// -fdebug-prefix-map (or -ffile-prefix-map) requires an absolute "to"
			// path (or it joins the path  with the working directory). Pick something
			// that makes sense for the target platform.
			var to string
			if cfg.BuildContext.GOOS == "windows" {
				to = filepath.Join(`\\_\_`, toPath)
			} else {
				to = filepath.Join("/_", toPath)
			}
			flags = append(slices.Clip(flags), prefixMapFlag+"="+from+"="+to)
		}
	}

	// Tell gcc to not insert truly random numbers into the build process
	// this ensures LTO won't create random numbers for symbols.
	if b.gccSupportsFlag(compiler, "-frandom-seed=1") {
		flags = append(flags, "-frandom-seed="+buildid.HashToString(a.actionID))
	}

	overlayPath := file
	if p, ok := a.nonGoOverlay[overlayPath]; ok {
		overlayPath = p
	}
	output, err := sh.runOut(filepath.Dir(overlayPath), b.cCompilerEnv(), compiler, flags, "-o", outfile, "-c", filepath.Base(overlayPath))

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
			return b.ccompile(a, outfile, newFlags, file, compiler)
		}
	}

	if len(output) > 0 && err == nil && os.Getenv("GO_BUILDER_NAME") != "" {
		output = append(output, "C compiler warning promoted to error on Go builders\n"...)
		err = errors.New("warning promoted to error")
	}

	return sh.reportCmd("", "", output, err)
}

// gccld runs the gcc linker to create an executable from a set of object files.
func (b *Builder) gccld(a *Action, objdir, outfile string, flags []string, objs []string) error {
	p := a.Package
	sh := b.Shell(a)
	var cmd []string
	if len(p.CXXFiles) > 0 || len(p.SwigCXXFiles) > 0 {
		cmd = b.GxxCmd(p.Dir, objdir)
	} else {
		cmd = b.GccCmd(p.Dir, objdir)
	}

	cmdargs := []any{cmd, "-o", outfile, objs, flags}
	_, err := sh.runOut(base.Cwd(), b.cCompilerEnv(), cmdargs...)

	// Note that failure is an expected outcome here, so we report output only
	// in debug mode and don't report the error.
	if cfg.BuildN || cfg.BuildX {
		saw := "succeeded"
		if err != nil {
			saw = "failed"
		}
		sh.ShowCmd("", "%s # test for internal linking errors (%s)", joinUnambiguously(str.StringList(cmdargs...)), saw)
	}

	return err
}

// GccCmd returns a gcc command line prefix
// defaultCC is defined in zdefaultcc.go, written by cmd/dist.
func (b *Builder) GccCmd(incdir, workdir string) []string {
	return b.compilerCmd(b.ccExe(), incdir, workdir)
}

// GxxCmd returns a g++ command line prefix
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
	return envList("CC", cfg.DefaultCC(cfg.Goos, cfg.Goarch))
}

// cxxExe returns the CXX compiler setting without all the extra flags we add implicitly.
func (b *Builder) cxxExe() []string {
	return envList("CXX", cfg.DefaultCXX(cfg.Goos, cfg.Goarch))
}

// fcExe returns the FC compiler setting without all the extra flags we add implicitly.
func (b *Builder) fcExe() []string {
	return envList("FC", "gfortran")
}

// compilerCmd returns a command line prefix for the given environment
// variable and using the default command when the variable is empty.
func (b *Builder) compilerCmd(compiler []string, incdir, workdir string) []string {
	a := append(compiler, "-I", incdir)

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

	if cfg.Goos == "aix" {
		// mcmodel=large must always be enabled to allow large TOC.
		a = append(a, "-mcmodel=large")
	}

	// disable ASCII art in clang errors, if possible
	if b.gccSupportsFlag(compiler, "-fno-caret-diagnostics") {
		a = append(a, "-fno-caret-diagnostics")
	}
	// clang is too smart about command-line arguments
	if b.gccSupportsFlag(compiler, "-Qunused-arguments") {
		a = append(a, "-Qunused-arguments")
	}

	// zig cc passes --gc-sections to the underlying linker, which then causes
	// undefined symbol errors when compiling with cgo but without C code.
	// https://github.com/golang/go/issues/52690
	if b.gccSupportsFlag(compiler, "-Wl,--no-gc-sections") {
		a = append(a, "-Wl,--no-gc-sections")
	}

	// disable word wrapping in error messages
	a = append(a, "-fmessage-length=0")

	// Tell gcc not to include the work directory in object files.
	if b.gccSupportsFlag(compiler, "-fdebug-prefix-map=a=b") {
		if workdir == "" {
			workdir = b.WorkDir
		}
		workdir = strings.TrimSuffix(workdir, string(filepath.Separator))
		if b.gccSupportsFlag(compiler, "-ffile-prefix-map=a=b") {
			a = append(a, "-ffile-prefix-map="+workdir+"=/tmp/go-build")
		} else {
			a = append(a, "-fdebug-prefix-map="+workdir+"=/tmp/go-build")
		}
	}

	// Tell gcc not to include flags in object files, which defeats the
	// point of -fdebug-prefix-map above.
	if b.gccSupportsFlag(compiler, "-gno-record-gcc-switches") {
		a = append(a, "-gno-record-gcc-switches")
	}

	// On OS X, some of the compilers behave as if -fno-common
	// is always set, and the Mach-O linker in 6l/8l assumes this.
	// See https://golang.org/issue/3253.
	if cfg.Goos == "darwin" || cfg.Goos == "ios" {
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
	// We use the background shell for operations here because, while this is
	// triggered by some Action, it's not really about that Action, and often we
	// just get the results from the global cache.
	sh := b.BackgroundShell()

	key := [2]string{compiler[0], flag}

	// We used to write an empty C file, but that gets complicated with go
	// build -n. We tried using a file that does not exist, but that fails on
	// systems with GCC version 4.2.1; that is the last GPLv2 version of GCC,
	// so some systems have frozen on it. Now we pass an empty file on stdin,
	// which should work at least for GCC and clang.
	//
	// If the argument is "-Wl,", then it is testing the linker. In that case,
	// skip "-c". If it's not "-Wl,", then we are testing the compiler and can
	// omit the linking step with "-c".
	//
	// Using the same CFLAGS/LDFLAGS here and for building the program.

	// On the iOS builder the command
	//   $CC -Wl,--no-gc-sections -x c - -o /dev/null < /dev/null
	// is failing with:
	//   Unable to remove existing file: Invalid argument
	tmp := os.DevNull
	if runtime.GOOS == "windows" || runtime.GOOS == "ios" {
		f, err := os.CreateTemp(b.WorkDir, "")
		if err != nil {
			return false
		}
		f.Close()
		tmp = f.Name()
		defer os.Remove(tmp)
	}

	cmdArgs := str.StringList(compiler, flag)
	if strings.HasPrefix(flag, "-Wl,") /* linker flag */ {
		ldflags, err := buildFlags("LDFLAGS", DefaultCFlags, nil, checkLinkerFlags)
		if err != nil {
			return false
		}
		cmdArgs = append(cmdArgs, ldflags...)
	} else { /* compiler flag, add "-c" */
		cflags, err := buildFlags("CFLAGS", DefaultCFlags, nil, checkCompilerFlags)
		if err != nil {
			return false
		}
		cmdArgs = append(cmdArgs, cflags...)
		cmdArgs = append(cmdArgs, "-c")
	}

	cmdArgs = append(cmdArgs, "-x", "c", "-", "-o", tmp)

	if cfg.BuildN {
		sh.ShowCmd(b.WorkDir, "%s || true", joinUnambiguously(cmdArgs))
		return false
	}

	// gccCompilerID acquires b.exec, so do before acquiring lock.
	compilerID, cacheOK := b.gccCompilerID(compiler[0])

	b.exec.Lock()
	defer b.exec.Unlock()
	if b, ok := b.flagCache[key]; ok {
		return b
	}
	if b.flagCache == nil {
		b.flagCache = make(map[[2]string]bool)
	}

	// Look in build cache.
	var flagID cache.ActionID
	if cacheOK {
		flagID = cache.Subkey(compilerID, "gccSupportsFlag "+flag)
		if data, _, err := cache.GetBytes(cache.Default(), flagID); err == nil {
			supported := string(data) == "true"
			b.flagCache[key] = supported
			return supported
		}
	}

	if cfg.BuildX {
		sh.ShowCmd(b.WorkDir, "%s || true", joinUnambiguously(cmdArgs))
	}
	cmd := exec.Command(cmdArgs[0], cmdArgs[1:]...)
	cmd.Dir = b.WorkDir
	cmd.Env = append(cmd.Environ(), "LC_ALL=C")
	out, _ := cmd.CombinedOutput()
	// GCC says "unrecognized command line option".
	// clang says "unknown argument".
	// tcc says "unsupported"
	// AIX says "not recognized"
	// Older versions of GCC say "unrecognised debug output level".
	// For -fsplit-stack GCC says "'-fsplit-stack' is not supported".
	supported := !bytes.Contains(out, []byte("unrecognized")) &&
		!bytes.Contains(out, []byte("unknown")) &&
		!bytes.Contains(out, []byte("unrecognised")) &&
		!bytes.Contains(out, []byte("is not supported")) &&
		!bytes.Contains(out, []byte("not recognized")) &&
		!bytes.Contains(out, []byte("unsupported"))

	if cacheOK {
		s := "false"
		if supported {
			s = "true"
		}
		cache.PutBytes(cache.Default(), flagID, []byte(s))
	}

	b.flagCache[key] = supported
	return supported
}

// statString returns a string form of an os.FileInfo, for serializing and comparison.
func statString(info os.FileInfo) string {
	return fmt.Sprintf("stat %d %x %v %v\n", info.Size(), uint64(info.Mode()), info.ModTime(), info.IsDir())
}

// gccCompilerID returns a build cache key for the current gcc,
// as identified by running 'compiler'.
// The caller can use subkeys of the key.
// Other parts of cmd/go can use the id as a hash
// of the installed compiler version.
func (b *Builder) gccCompilerID(compiler string) (id cache.ActionID, ok bool) {
	// We use the background shell for operations here because, while this is
	// triggered by some Action, it's not really about that Action, and often we
	// just get the results from the global cache.
	sh := b.BackgroundShell()

	if cfg.BuildN {
		sh.ShowCmd(b.WorkDir, "%s || true", joinUnambiguously([]string{compiler, "--version"}))
		return cache.ActionID{}, false
	}

	b.exec.Lock()
	defer b.exec.Unlock()

	if id, ok := b.gccCompilerIDCache[compiler]; ok {
		return id, ok
	}

	// We hash the compiler's full path to get a cache entry key.
	// That cache entry holds a validation description,
	// which is of the form:
	//
	//	filename \x00 statinfo \x00
	//	...
	//	compiler id
	//
	// If os.Stat of each filename matches statinfo,
	// then the entry is still valid, and we can use the
	// compiler id without any further expense.
	//
	// Otherwise, we compute a new validation description
	// and compiler id (below).
	exe, err := pathcache.LookPath(compiler)
	if err != nil {
		return cache.ActionID{}, false
	}

	h := cache.NewHash("gccCompilerID")
	fmt.Fprintf(h, "gccCompilerID %q", exe)
	key := h.Sum()
	data, _, err := cache.GetBytes(cache.Default(), key)
	if err == nil && len(data) > len(id) {
		stats := strings.Split(string(data[:len(data)-len(id)]), "\x00")
		if len(stats)%2 != 0 {
			goto Miss
		}
		for i := 0; i+2 <= len(stats); i++ {
			info, err := os.Stat(stats[i])
			if err != nil || statString(info) != stats[i+1] {
				goto Miss
			}
		}
		copy(id[:], data[len(data)-len(id):])
		return id, true
	Miss:
	}

	// Validation failed. Compute a new description (in buf) and compiler ID (in h).
	// For now, there are only at most two filenames in the stat information.
	// The first one is the compiler executable we invoke.
	// The second is the underlying compiler as reported by -v -###
	// (see b.gccToolID implementation in buildid.go).
	toolID, exe2, err := b.gccToolID(compiler, "c")
	if err != nil {
		return cache.ActionID{}, false
	}

	exes := []string{exe, exe2}
	str.Uniq(&exes)
	fmt.Fprintf(h, "gccCompilerID %q %q\n", exes, toolID)
	id = h.Sum()

	var buf bytes.Buffer
	for _, exe := range exes {
		if exe == "" {
			continue
		}
		info, err := os.Stat(exe)
		if err != nil {
			return cache.ActionID{}, false
		}
		buf.WriteString(exe)
		buf.WriteString("\x00")
		buf.WriteString(statString(info))
		buf.WriteString("\x00")
	}
	buf.Write(id[:])

	cache.PutBytes(cache.Default(), key, buf.Bytes())
	if b.gccCompilerIDCache == nil {
		b.gccCompilerIDCache = make(map[string]cache.ActionID)
	}
	b.gccCompilerIDCache[compiler] = id
	return id, true
}

// gccArchArgs returns arguments to pass to gcc based on the architecture.
func (b *Builder) gccArchArgs() []string {
	switch cfg.Goarch {
	case "386":
		return []string{"-m32"}
	case "amd64":
		if cfg.Goos == "darwin" {
			return []string{"-arch", "x86_64", "-m64"}
		}
		return []string{"-m64"}
	case "arm64":
		if cfg.Goos == "darwin" {
			return []string{"-arch", "arm64"}
		}
	case "arm":
		return []string{"-marm"} // not thumb
	case "s390x":
		// minimum supported s390x version on Go is z13
		return []string{"-m64", "-march=z13"}
	case "mips64", "mips64le":
		args := []string{"-mabi=64"}
		if cfg.GOMIPS64 == "hardfloat" {
			return append(args, "-mhard-float")
		} else if cfg.GOMIPS64 == "softfloat" {
			return append(args, "-msoft-float")
		}
	case "mips", "mipsle":
		args := []string{"-mabi=32", "-march=mips32"}
		if cfg.GOMIPS == "hardfloat" {
			return append(args, "-mhard-float", "-mfp32", "-mno-odd-spreg")
		} else if cfg.GOMIPS == "softfloat" {
			return append(args, "-msoft-float")
		}
	case "loong64":
		return []string{"-mabi=lp64d"}
	case "ppc64":
		if cfg.Goos == "aix" {
			return []string{"-maix64"}
		}
	}
	return nil
}

// envList returns the value of the given environment variable broken
// into fields, using the default value when the variable is empty.
//
// The environment variable must be quoted correctly for
// quoted.Split. This should be done before building
// anything, for example, in BuildInit.
func envList(key, def string) []string {
	v := cfg.Getenv(key)
	if v == "" {
		v = def
	}
	args, err := quoted.Split(v)
	if err != nil {
		panic(fmt.Sprintf("could not parse environment variable %s with value %q: %v", key, v, err))
	}
	return args
}

// CFlags returns the flags to use when invoking the C, C++ or Fortran compilers, or cgo.
func (b *Builder) CFlags(p *load.Package) (cppflags, cflags, cxxflags, fflags, ldflags []string, err error) {
	if cppflags, err = buildFlags("CPPFLAGS", "", p.CgoCPPFLAGS, checkCompilerFlags); err != nil {
		return
	}
	if cflags, err = buildFlags("CFLAGS", DefaultCFlags, p.CgoCFLAGS, checkCompilerFlags); err != nil {
		return
	}
	if cxxflags, err = buildFlags("CXXFLAGS", DefaultCFlags, p.CgoCXXFLAGS, checkCompilerFlags); err != nil {
		return
	}
	if fflags, err = buildFlags("FFLAGS", DefaultCFlags, p.CgoFFLAGS, checkCompilerFlags); err != nil {
		return
	}
	if ldflags, err = buildFlags("LDFLAGS", DefaultCFlags, p.CgoLDFLAGS, checkLinkerFlags); err != nil {
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

var cgoRe = lazyregexp.New(`[/\\:]`)

func (b *Builder) cgo(a *Action, cgoExe, objdir string, pcCFLAGS, pcLDFLAGS, cgofiles, gccfiles, gxxfiles, mfiles, ffiles []string) (outGo, outObj []string, err error) {
	p := a.Package
	sh := b.Shell(a)

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
		fc := cfg.Getenv("FC")
		if fc == "" {
			fc = "gfortran"
		}
		if strings.Contains(fc, "gfortran") {
			cgoLDFLAGS = append(cgoLDFLAGS, "-lgfortran")
		}
	}

	// Scrutinize CFLAGS and related for flags that might cause
	// problems if we are using internal linking (for example, use of
	// plugins, LTO, etc) by calling a helper routine that builds on
	// the existing CGO flags allow-lists. If we see anything
	// suspicious, emit a special token file "preferlinkext" (known to
	// the linker) in the object file to signal the that it should not
	// try to link internally and should revert to external linking.
	// The token we pass is a suggestion, not a mandate; if a user is
	// explicitly asking for a specific linkmode via the "-linkmode"
	// flag, the token will be ignored. NB: in theory we could ditch
	// the token approach and just pass a flag to the linker when we
	// eventually invoke it, and the linker flag could then be
	// documented (although coming up with a simple explanation of the
	// flag might be challenging). For more context see issues #58619,
	// #58620, and #58848.
	flagSources := []string{"CGO_CFLAGS", "CGO_CXXFLAGS", "CGO_FFLAGS"}
	flagLists := [][]string{cgoCFLAGS, cgoCXXFLAGS, cgoFFLAGS}
	if flagsNotCompatibleWithInternalLinking(flagSources, flagLists) {
		tokenFile := objdir + "preferlinkext"
		if err := sh.writeFile(tokenFile, nil); err != nil {
			return nil, nil, err
		}
		outObj = append(outObj, tokenFile)
	}

	if cfg.BuildMSan {
		cgoCFLAGS = append([]string{"-fsanitize=memory"}, cgoCFLAGS...)
		cgoLDFLAGS = append([]string{"-fsanitize=memory"}, cgoLDFLAGS...)
	}
	if cfg.BuildASan {
		cgoCFLAGS = append([]string{"-fsanitize=address"}, cgoCFLAGS...)
		cgoLDFLAGS = append([]string{"-fsanitize=address"}, cgoLDFLAGS...)
	}

	// Allows including _cgo_export.h, as well as the user's .h files,
	// from .[ch] files in the package.
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
	if p.Standard && (p.ImportPath == "runtime/race" || p.ImportPath == "runtime/msan" || p.ImportPath == "runtime/cgo" || p.ImportPath == "runtime/asan") {
		cgoflags = append(cgoflags, "-import_syscall=false")
	}

	// cgoLDFLAGS, which includes p.CgoLDFLAGS, can be very long.
	// Pass it to cgo on the command line, so that we use a
	// response file if necessary.
	//
	// These flags are recorded in the generated _cgo_gotypes.go file
	// using //go:cgo_ldflag directives, the compiler records them in the
	// object file for the package, and then the Go linker passes them
	// along to the host linker. At this point in the code, cgoLDFLAGS
	// consists of the original $CGO_LDFLAGS (unchecked) and all the
	// flags put together from source code (checked).
	cgoenv := b.cCompilerEnv()
	cgoenv = append(cgoenv, cfgChangedEnv...)
	var ldflagsOption []string
	if len(cgoLDFLAGS) > 0 {
		flags := make([]string, len(cgoLDFLAGS))
		for i, f := range cgoLDFLAGS {
			flags[i] = strconv.Quote(f)
		}
		ldflagsOption = []string{"-ldflags=" + strings.Join(flags, " ")}

		// Remove CGO_LDFLAGS from the environment.
		cgoenv = append(cgoenv, "CGO_LDFLAGS=")
	}

	if cfg.BuildToolchainName == "gccgo" {
		if b.gccSupportsFlag([]string{BuildToolchain.compiler()}, "-fsplit-stack") {
			cgoCFLAGS = append(cgoCFLAGS, "-fsplit-stack")
		}
		cgoflags = append(cgoflags, "-gccgo")
		if pkgpath := gccgoPkgpath(p); pkgpath != "" {
			cgoflags = append(cgoflags, "-gccgopkgpath="+pkgpath)
		}
		if !BuildToolchain.(gccgoToolchain).supportsCgoIncomplete(b, a) {
			cgoflags = append(cgoflags, "-gccgo_define_cgoincomplete")
		}
	}

	switch cfg.BuildBuildmode {
	case "c-archive", "c-shared":
		// Tell cgo that if there are any exported functions
		// it should generate a header file that C code can
		// #include.
		cgoflags = append(cgoflags, "-exportheader="+objdir+"_cgo_install.h")
	}

	// Rewrite overlaid paths in cgo files.
	// cgo adds //line and #line pragmas in generated files with these paths.
	var trimpath []string
	for i := range cgofiles {
		path := mkAbs(p.Dir, cgofiles[i])
		if fsys.Replaced(path) {
			actual := fsys.Actual(path)
			cgofiles[i] = actual
			trimpath = append(trimpath, actual+"=>"+path)
		}
	}
	if len(trimpath) > 0 {
		cgoflags = append(cgoflags, "-trimpath", strings.Join(trimpath, ";"))
	}

	if err := sh.run(p.Dir, p.ImportPath, cgoenv, cfg.BuildToolexec, cgoExe, "-objdir", objdir, "-importpath", p.ImportPath, cgoflags, ldflagsOption, "--", cgoCPPFLAGS, cgoCFLAGS, cgofiles); err != nil {
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
		if err := b.gcc(a, a.Objdir, ofile, cflags, objdir+cfile); err != nil {
			return nil, nil, err
		}
		outObj = append(outObj, ofile)
	}

	for _, file := range gccfiles {
		ofile := nextOfile()
		if err := b.gcc(a, a.Objdir, ofile, cflags, file); err != nil {
			return nil, nil, err
		}
		outObj = append(outObj, ofile)
	}

	cxxflags := str.StringList(cgoCPPFLAGS, cgoCXXFLAGS)
	for _, file := range gxxfiles {
		ofile := nextOfile()
		if err := b.gxx(a, a.Objdir, ofile, cxxflags, file); err != nil {
			return nil, nil, err
		}
		outObj = append(outObj, ofile)
	}

	for _, file := range mfiles {
		ofile := nextOfile()
		if err := b.gcc(a, a.Objdir, ofile, cflags, file); err != nil {
			return nil, nil, err
		}
		outObj = append(outObj, ofile)
	}

	fflags := str.StringList(cgoCPPFLAGS, cgoFFLAGS)
	for _, file := range ffiles {
		ofile := nextOfile()
		if err := b.gfortran(a, a.Objdir, ofile, fflags, file); err != nil {
			return nil, nil, err
		}
		outObj = append(outObj, ofile)
	}

	switch cfg.BuildToolchainName {
	case "gc":
		importGo := objdir + "_cgo_import.go"
		dynOutGo, dynOutObj, err := b.dynimport(a, objdir, importGo, cgoExe, cflags, cgoLDFLAGS, outObj)
		if err != nil {
			return nil, nil, err
		}
		if dynOutGo != "" {
			outGo = append(outGo, dynOutGo)
		}
		if dynOutObj != "" {
			outObj = append(outObj, dynOutObj)
		}

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

	// Double check the //go:cgo_ldflag comments in the generated files.
	// The compiler only permits such comments in files whose base name
	// starts with "_cgo_". Make sure that the comments in those files
	// are safe. This is a backstop against people somehow smuggling
	// such a comment into a file generated by cgo.
	if cfg.BuildToolchainName == "gc" && !cfg.BuildN {
		var flags []string
		for _, f := range outGo {
			if !strings.HasPrefix(filepath.Base(f), "_cgo_") {
				continue
			}

			src, err := os.ReadFile(f)
			if err != nil {
				return nil, nil, err
			}

			const cgoLdflag = "//go:cgo_ldflag"
			idx := bytes.Index(src, []byte(cgoLdflag))
			for idx >= 0 {
				// We are looking at //go:cgo_ldflag.
				// Find start of line.
				start := bytes.LastIndex(src[:idx], []byte("\n"))
				if start == -1 {
					start = 0
				}

				// Find end of line.
				end := bytes.Index(src[idx:], []byte("\n"))
				if end == -1 {
					end = len(src)
				} else {
					end += idx
				}

				// Check for first line comment in line.
				// We don't worry about /* */ comments,
				// which normally won't appear in files
				// generated by cgo.
				commentStart := bytes.Index(src[start:], []byte("//"))
				commentStart += start
				// If that line comment is //go:cgo_ldflag,
				// it's a match.
				if bytes.HasPrefix(src[commentStart:], []byte(cgoLdflag)) {
					// Pull out the flag, and unquote it.
					// This is what the compiler does.
					flag := string(src[idx+len(cgoLdflag) : end])
					flag = strings.TrimSpace(flag)
					flag = strings.Trim(flag, `"`)
					flags = append(flags, flag)
				}
				src = src[end:]
				idx = bytes.Index(src, []byte(cgoLdflag))
			}
		}

		// We expect to find the contents of cgoLDFLAGS in flags.
		if len(cgoLDFLAGS) > 0 {
		outer:
			for i := range flags {
				for j, f := range cgoLDFLAGS {
					if f != flags[i+j] {
						continue outer
					}
				}
				flags = append(flags[:i], flags[i+len(cgoLDFLAGS):]...)
				break
			}
		}

		if err := checkLinkerFlags("LDFLAGS", "go:cgo_ldflag", flags); err != nil {
			return nil, nil, err
		}
	}

	return outGo, outObj, nil
}

// flagsNotCompatibleWithInternalLinking scans the list of cgo
// compiler flags (C/C++/Fortran) looking for flags that might cause
// problems if the build in question uses internal linking. The
// primary culprits are use of plugins or use of LTO, but we err on
// the side of caution, supporting only those flags that are on the
// allow-list for safe flags from security perspective. Return is TRUE
// if a sensitive flag is found, FALSE otherwise.
func flagsNotCompatibleWithInternalLinking(sourceList []string, flagListList [][]string) bool {
	for i := range sourceList {
		sn := sourceList[i]
		fll := flagListList[i]
		if err := checkCompilerFlagsForInternalLink(sn, sn, fll); err != nil {
			return true
		}
	}
	return false
}

// dynimport creates a Go source file named importGo containing
// //go:cgo_import_dynamic directives for each symbol or library
// dynamically imported by the object files outObj.
// dynOutGo, if not empty, is a new Go file to build as part of the package.
// dynOutObj, if not empty, is a new file to add to the generated archive.
func (b *Builder) dynimport(a *Action, objdir, importGo, cgoExe string, cflags, cgoLDFLAGS, outObj []string) (dynOutGo, dynOutObj string, err error) {
	p := a.Package
	sh := b.Shell(a)

	cfile := objdir + "_cgo_main.c"
	ofile := objdir + "_cgo_main.o"
	if err := b.gcc(a, objdir, ofile, cflags, cfile); err != nil {
		return "", "", err
	}

	// Gather .syso files from this package and all (transitive) dependencies.
	var syso []string
	seen := make(map[*Action]bool)
	var gatherSyso func(*Action)
	gatherSyso = func(a1 *Action) {
		if seen[a1] {
			return
		}
		seen[a1] = true
		if p1 := a1.Package; p1 != nil {
			syso = append(syso, mkAbsFiles(p1.Dir, p1.SysoFiles)...)
		}
		for _, a2 := range a1.Deps {
			gatherSyso(a2)
		}
	}
	gatherSyso(a)
	sort.Strings(syso)
	str.Uniq(&syso)
	linkobj := str.StringList(ofile, outObj, syso)
	dynobj := objdir + "_cgo_.o"

	ldflags := cgoLDFLAGS
	if (cfg.Goarch == "arm" && cfg.Goos == "linux") || cfg.Goos == "android" {
		if !slices.Contains(ldflags, "-no-pie") {
			// we need to use -pie for Linux/ARM to get accurate imported sym (added in https://golang.org/cl/5989058)
			// this seems to be outdated, but we don't want to break existing builds depending on this (Issue 45940)
			ldflags = append(ldflags, "-pie")
		}
		if slices.Contains(ldflags, "-pie") && slices.Contains(ldflags, "-static") {
			// -static -pie doesn't make sense, and causes link errors.
			// Issue 26197.
			n := make([]string, 0, len(ldflags)-1)
			for _, flag := range ldflags {
				if flag != "-static" {
					n = append(n, flag)
				}
			}
			ldflags = n
		}
	}
	if err := b.gccld(a, objdir, dynobj, ldflags, linkobj); err != nil {
		// We only need this information for internal linking.
		// If this link fails, mark the object as requiring
		// external linking. This link can fail for things like
		// syso files that have unexpected dependencies.
		// cmd/link explicitly looks for the name "dynimportfail".
		// See issue #52863.
		fail := objdir + "dynimportfail"
		if err := sh.writeFile(fail, nil); err != nil {
			return "", "", err
		}
		return "", fail, nil
	}

	// cgo -dynimport
	var cgoflags []string
	if p.Standard && p.ImportPath == "runtime/cgo" {
		cgoflags = []string{"-dynlinker"} // record path to dynamic linker
	}
	err = sh.run(base.Cwd(), p.ImportPath, b.cCompilerEnv(), cfg.BuildToolexec, cgoExe, "-dynpackage", p.Name, "-dynimport", dynobj, "-dynout", importGo, cgoflags)
	if err != nil {
		return "", "", err
	}
	return importGo, "", nil
}

// Run SWIG on all SWIG input files.
// TODO: Don't build a shared library, once SWIG emits the necessary
// pragmas for external linking.
func (b *Builder) swig(a *Action, objdir string, pcCFLAGS []string) (outGo, outC, outCXX []string, err error) {
	p := a.Package

	if err := b.swigVersionCheck(); err != nil {
		return nil, nil, nil, err
	}

	intgosize, err := b.swigIntSize(objdir)
	if err != nil {
		return nil, nil, nil, err
	}

	for _, f := range p.SwigFiles {
		goFile, cFile, err := b.swigOne(a, f, objdir, pcCFLAGS, false, intgosize)
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
		goFile, cxxFile, err := b.swigOne(a, f, objdir, pcCFLAGS, true, intgosize)
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
	sh := b.BackgroundShell()
	out, err := sh.runOut(".", nil, "swig", "-version")
	if err != nil {
		return err
	}
	re := regexp.MustCompile(`[vV]ersion +(\d+)([.]\d+)?([.]\d+)?`)
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
	if err = os.WriteFile(src, []byte(swigIntSizeCode), 0666); err != nil {
		return
	}
	srcs := []string{src}

	p := load.GoFilesPackage(context.TODO(), load.PackageOpts{}, srcs)

	if _, _, e := BuildToolchain.gc(b, &Action{Mode: "swigDoIntSize", Package: p, Objdir: objdir}, "", nil, nil, "", false, "", srcs); e != nil {
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
func (b *Builder) swigOne(a *Action, file, objdir string, pcCFLAGS []string, cxx bool, intgosize string) (outGo, outC string, err error) {
	p := a.Package
	sh := b.Shell(a)

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

	out, err := sh.runOut(p.Dir, nil, "swig", args, file)
	if err != nil && (bytes.Contains(out, []byte("-intgosize")) || bytes.Contains(out, []byte("-cgo"))) {
		return "", "", errors.New("must have SWIG version >= 3.0.6")
	}
	if err := sh.reportCmd("", "", out, err); err != nil {
		return "", "", err
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
	if cfg.BuildX || cfg.BuildN {
		sh.ShowCmd("", "mv %s %s", goFile, newGoFile)
	}
	if !cfg.BuildN {
		if err := os.Rename(goFile, newGoFile); err != nil {
			return "", "", err
		}
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

// actualFiles applies fsys.Actual to the list of files.
func actualFiles(files []string) []string {
	a := make([]string, len(files))
	for i, f := range files {
		a[i] = fsys.Actual(f)
	}
	return a
}

// passLongArgsInResponseFiles modifies cmd such that, for
// certain programs, long arguments are passed in "response files", a
// file on disk with the arguments, with one arg per line. An actual
// argument starting with '@' means that the rest of the argument is
// a filename of arguments to expand.
//
// See issues 18468 (Windows) and 37768 (Darwin).
func passLongArgsInResponseFiles(cmd *exec.Cmd) (cleanup func()) {
	cleanup = func() {} // no cleanup by default

	var argLen int
	for _, arg := range cmd.Args {
		argLen += len(arg)
	}

	// If we're not approaching 32KB of args, just pass args normally.
	// (use 30KB instead to be conservative; not sure how accounting is done)
	if !useResponseFile(cmd.Path, argLen) {
		return
	}

	tf, err := os.CreateTemp("", "args")
	if err != nil {
		log.Fatalf("error writing long arguments to response file: %v", err)
	}
	cleanup = func() { os.Remove(tf.Name()) }
	var buf bytes.Buffer
	for _, arg := range cmd.Args[1:] {
		fmt.Fprintf(&buf, "%s\n", encodeArg(arg))
	}
	if _, err := tf.Write(buf.Bytes()); err != nil {
		tf.Close()
		cleanup()
		log.Fatalf("error writing long arguments to response file: %v", err)
	}
	if err := tf.Close(); err != nil {
		cleanup()
		log.Fatalf("error writing long arguments to response file: %v", err)
	}
	cmd.Args = []string{cmd.Args[0], "@" + tf.Name()}
	return cleanup
}

func useResponseFile(path string, argLen int) bool {
	// Unless the program uses objabi.Flagparse, which understands
	// response files, don't use response files.
	// TODO: Note that other toolchains like CC are missing here for now.
	prog := strings.TrimSuffix(filepath.Base(path), ".exe")
	switch prog {
	case "compile", "link", "cgo", "asm", "cover":
	default:
		return false
	}

	if argLen > sys.ExecArgLengthLimit {
		return true
	}

	// On the Go build system, use response files about 10% of the
	// time, just to exercise this codepath.
	isBuilder := os.Getenv("GO_BUILDER_NAME") != ""
	if isBuilder && rand.Intn(10) == 0 {
		return true
	}

	return false
}

// encodeArg encodes an argument for response file writing.
func encodeArg(arg string) string {
	// If there aren't any characters we need to reencode, fastpath out.
	if !strings.ContainsAny(arg, "\\\n") {
		return arg
	}
	var b strings.Builder
	for _, r := range arg {
		switch r {
		case '\\':
			b.WriteByte('\\')
			b.WriteByte('\\')
		case '\n':
			b.WriteByte('\\')
			b.WriteByte('n')
		default:
			b.WriteRune(r)
		}
	}
	return b.String()
}
