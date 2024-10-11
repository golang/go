// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Action graph creation (planning).

package work

import (
	"bufio"
	"bytes"
	"cmd/internal/cov/covcmd"
	"container/heap"
	"context"
	"debug/elf"
	"encoding/json"
	"fmt"
	"internal/platform"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"cmd/go/internal/base"
	"cmd/go/internal/cache"
	"cmd/go/internal/cfg"
	"cmd/go/internal/load"
	"cmd/go/internal/str"
	"cmd/go/internal/trace"
	"cmd/internal/buildid"
	"cmd/internal/robustio"
)

// A Builder holds global state about a build.
// It does not hold per-package state, because we
// build packages in parallel, and the builder is shared.
type Builder struct {
	WorkDir            string                    // the temporary work directory (ends in filepath.Separator)
	actionCache        map[cacheKey]*Action      // a cache of already-constructed actions
	flagCache          map[[2]string]bool        // a cache of supported compiler flags
	gccCompilerIDCache map[string]cache.ActionID // cache for gccCompilerID

	IsCmdList           bool // running as part of go list; set p.Stale and additional fields below
	NeedError           bool // list needs p.Error
	NeedExport          bool // list needs p.Export
	NeedCompiledGoFiles bool // list needs p.CompiledGoFiles
	AllowErrors         bool // errors don't immediately exit the program

	objdirSeq int // counter for NewObjdir
	pkgSeq    int

	backgroundSh *Shell // Shell that per-Action Shells are derived from

	exec      sync.Mutex
	readySema chan bool
	ready     actionQueue

	id           sync.Mutex
	toolIDCache  map[string]string // tool name -> tool ID
	buildIDCache map[string]string // file name -> build ID
}

// NOTE: Much of Action would not need to be exported if not for test.
// Maybe test functionality should move into this package too?

// An Actor runs an action.
type Actor interface {
	Act(*Builder, context.Context, *Action) error
}

// An ActorFunc is an Actor that calls the function.
type ActorFunc func(*Builder, context.Context, *Action) error

func (f ActorFunc) Act(b *Builder, ctx context.Context, a *Action) error {
	return f(b, ctx, a)
}

// An Action represents a single action in the action graph.
type Action struct {
	Mode       string        // description of action operation
	Package    *load.Package // the package this action works on
	Deps       []*Action     // actions that must happen before this one
	Actor      Actor         // the action itself (nil = no-op)
	IgnoreFail bool          // whether to run f even if dependencies fail
	TestOutput *bytes.Buffer // test output buffer
	Args       []string      // additional args for runProgram

	triggers []*Action // inverse of deps

	buggyInstall bool // is this a buggy install (see -linkshared)?

	TryCache func(*Builder, *Action) bool // callback for cache bypass

	// Generated files, directories.
	Objdir   string         // directory for intermediate objects
	Target   string         // goal of the action: the created package or executable
	built    string         // the actual created package or executable
	actionID cache.ActionID // cache ID of action input
	buildID  string         // build ID of action output

	VetxOnly  bool       // Mode=="vet": only being called to supply info about dependencies
	needVet   bool       // Mode=="build": need to fill in vet config
	needBuild bool       // Mode=="build": need to do actual build (can be false if needVet is true)
	vetCfg    *vetConfig // vet config
	output    []byte     // output redirect buffer (nil means use b.Print)

	sh *Shell // lazily created per-Action shell; see Builder.Shell

	// Execution state.
	pending      int               // number of deps yet to complete
	priority     int               // relative execution priority
	Failed       bool              // whether the action failed
	json         *actionJSON       // action graph information
	nonGoOverlay map[string]string // map from non-.go source files to copied files in objdir. Nil if no overlay is used.
	traceSpan    *trace.Span
}

// BuildActionID returns the action ID section of a's build ID.
func (a *Action) BuildActionID() string { return actionID(a.buildID) }

// BuildContentID returns the content ID section of a's build ID.
func (a *Action) BuildContentID() string { return contentID(a.buildID) }

// BuildID returns a's build ID.
func (a *Action) BuildID() string { return a.buildID }

// BuiltTarget returns the actual file that was built. This differs
// from Target when the result was cached.
func (a *Action) BuiltTarget() string { return a.built }

// An actionQueue is a priority queue of actions.
type actionQueue []*Action

// Implement heap.Interface
func (q *actionQueue) Len() int           { return len(*q) }
func (q *actionQueue) Swap(i, j int)      { (*q)[i], (*q)[j] = (*q)[j], (*q)[i] }
func (q *actionQueue) Less(i, j int) bool { return (*q)[i].priority < (*q)[j].priority }
func (q *actionQueue) Push(x any)         { *q = append(*q, x.(*Action)) }
func (q *actionQueue) Pop() any {
	n := len(*q) - 1
	x := (*q)[n]
	*q = (*q)[:n]
	return x
}

func (q *actionQueue) push(a *Action) {
	if a.json != nil {
		a.json.TimeReady = time.Now()
	}
	heap.Push(q, a)
}

func (q *actionQueue) pop() *Action {
	return heap.Pop(q).(*Action)
}

type actionJSON struct {
	ID         int
	Mode       string
	Package    string
	Deps       []int     `json:",omitempty"`
	IgnoreFail bool      `json:",omitempty"`
	Args       []string  `json:",omitempty"`
	Link       bool      `json:",omitempty"`
	Objdir     string    `json:",omitempty"`
	Target     string    `json:",omitempty"`
	Priority   int       `json:",omitempty"`
	Failed     bool      `json:",omitempty"`
	Built      string    `json:",omitempty"`
	VetxOnly   bool      `json:",omitempty"`
	NeedVet    bool      `json:",omitempty"`
	NeedBuild  bool      `json:",omitempty"`
	ActionID   string    `json:",omitempty"`
	BuildID    string    `json:",omitempty"`
	TimeReady  time.Time `json:",omitempty"`
	TimeStart  time.Time `json:",omitempty"`
	TimeDone   time.Time `json:",omitempty"`

	Cmd     []string      // `json:",omitempty"`
	CmdReal time.Duration `json:",omitempty"`
	CmdUser time.Duration `json:",omitempty"`
	CmdSys  time.Duration `json:",omitempty"`
}

// cacheKey is the key for the action cache.
type cacheKey struct {
	mode string
	p    *load.Package
}

func actionGraphJSON(a *Action) string {
	var workq []*Action
	var inWorkq = make(map[*Action]int)

	add := func(a *Action) {
		if _, ok := inWorkq[a]; ok {
			return
		}
		inWorkq[a] = len(workq)
		workq = append(workq, a)
	}
	add(a)

	for i := 0; i < len(workq); i++ {
		for _, dep := range workq[i].Deps {
			add(dep)
		}
	}

	list := make([]*actionJSON, 0, len(workq))
	for id, a := range workq {
		if a.json == nil {
			a.json = &actionJSON{
				Mode:       a.Mode,
				ID:         id,
				IgnoreFail: a.IgnoreFail,
				Args:       a.Args,
				Objdir:     a.Objdir,
				Target:     a.Target,
				Failed:     a.Failed,
				Priority:   a.priority,
				Built:      a.built,
				VetxOnly:   a.VetxOnly,
				NeedBuild:  a.needBuild,
				NeedVet:    a.needVet,
			}
			if a.Package != nil {
				// TODO(rsc): Make this a unique key for a.Package somehow.
				a.json.Package = a.Package.ImportPath
			}
			for _, a1 := range a.Deps {
				a.json.Deps = append(a.json.Deps, inWorkq[a1])
			}
		}
		list = append(list, a.json)
	}

	js, err := json.MarshalIndent(list, "", "\t")
	if err != nil {
		fmt.Fprintf(os.Stderr, "go: writing debug action graph: %v\n", err)
		return ""
	}
	return string(js)
}

// BuildMode specifies the build mode:
// are we just building things or also installing the results?
type BuildMode int

const (
	ModeBuild BuildMode = iota
	ModeInstall
	ModeBuggyInstall

	ModeVetOnly = 1 << 8
)

// NewBuilder returns a new Builder ready for use.
//
// If workDir is the empty string, NewBuilder creates a WorkDir if needed
// and arranges for it to be removed in case of an unclean exit.
// The caller must Close the builder explicitly to clean up the WorkDir
// before a clean exit.
func NewBuilder(workDir string) *Builder {
	b := new(Builder)

	b.actionCache = make(map[cacheKey]*Action)
	b.toolIDCache = make(map[string]string)
	b.buildIDCache = make(map[string]string)

	if workDir != "" {
		b.WorkDir = workDir
	} else if cfg.BuildN {
		b.WorkDir = "$WORK"
	} else {
		if !buildInitStarted {
			panic("internal error: NewBuilder called before BuildInit")
		}
		tmp, err := os.MkdirTemp(cfg.Getenv("GOTMPDIR"), "go-build")
		if err != nil {
			base.Fatalf("go: creating work dir: %v", err)
		}
		if !filepath.IsAbs(tmp) {
			abs, err := filepath.Abs(tmp)
			if err != nil {
				os.RemoveAll(tmp)
				base.Fatalf("go: creating work dir: %v", err)
			}
			tmp = abs
		}
		b.WorkDir = tmp
		builderWorkDirs.Store(b, b.WorkDir)
		if cfg.BuildX || cfg.BuildWork {
			fmt.Fprintf(os.Stderr, "WORK=%s\n", b.WorkDir)
		}
	}

	b.backgroundSh = NewShell(b.WorkDir, nil)

	if err := CheckGOOSARCHPair(cfg.Goos, cfg.Goarch); err != nil {
		fmt.Fprintf(os.Stderr, "go: %v\n", err)
		base.SetExitStatus(2)
		base.Exit()
	}

	for _, tag := range cfg.BuildContext.BuildTags {
		if strings.Contains(tag, ",") {
			fmt.Fprintf(os.Stderr, "go: -tags space-separated list contains comma\n")
			base.SetExitStatus(2)
			base.Exit()
		}
	}

	return b
}

var builderWorkDirs sync.Map // *Builder → WorkDir

func (b *Builder) Close() error {
	wd, ok := builderWorkDirs.Load(b)
	if !ok {
		return nil
	}
	defer builderWorkDirs.Delete(b)

	if b.WorkDir != wd.(string) {
		base.Errorf("go: internal error: Builder WorkDir unexpectedly changed from %s to %s", wd, b.WorkDir)
	}

	if !cfg.BuildWork {
		if err := robustio.RemoveAll(b.WorkDir); err != nil {
			return err
		}
	}
	b.WorkDir = ""
	return nil
}

func closeBuilders() {
	leakedBuilders := 0
	builderWorkDirs.Range(func(bi, _ any) bool {
		leakedBuilders++
		if err := bi.(*Builder).Close(); err != nil {
			base.Error(err)
		}
		return true
	})

	if leakedBuilders > 0 && base.GetExitStatus() == 0 {
		fmt.Fprintf(os.Stderr, "go: internal error: Builder leaked on successful exit\n")
		base.SetExitStatus(1)
	}
}

func CheckGOOSARCHPair(goos, goarch string) error {
	if !platform.BuildModeSupported(cfg.BuildContext.Compiler, "default", goos, goarch) {
		return fmt.Errorf("unsupported GOOS/GOARCH pair %s/%s", goos, goarch)
	}
	return nil
}

// NewObjdir returns the name of a fresh object directory under b.WorkDir.
// It is up to the caller to call b.Mkdir on the result at an appropriate time.
// The result ends in a slash, so that file names in that directory
// can be constructed with direct string addition.
//
// NewObjdir must be called only from a single goroutine at a time,
// so it is safe to call during action graph construction, but it must not
// be called during action graph execution.
func (b *Builder) NewObjdir() string {
	b.objdirSeq++
	return str.WithFilePathSeparator(filepath.Join(b.WorkDir, fmt.Sprintf("b%03d", b.objdirSeq)))
}

// readpkglist returns the list of packages that were built into the shared library
// at shlibpath. For the native toolchain this list is stored, newline separated, in
// an ELF note with name "Go\x00\x00" and type 1. For GCCGO it is extracted from the
// .go_export section.
func readpkglist(shlibpath string) (pkgs []*load.Package) {
	var stk load.ImportStack
	if cfg.BuildToolchainName == "gccgo" {
		f, err := elf.Open(shlibpath)
		if err != nil {
			base.Fatal(fmt.Errorf("failed to open shared library: %v", err))
		}
		defer f.Close()
		sect := f.Section(".go_export")
		if sect == nil {
			base.Fatal(fmt.Errorf("%s: missing .go_export section", shlibpath))
		}
		data, err := sect.Data()
		if err != nil {
			base.Fatal(fmt.Errorf("%s: failed to read .go_export section: %v", shlibpath, err))
		}
		pkgpath := []byte("pkgpath ")
		for _, line := range bytes.Split(data, []byte{'\n'}) {
			if path, found := bytes.CutPrefix(line, pkgpath); found {
				path = bytes.TrimSuffix(path, []byte{';'})
				pkgs = append(pkgs, load.LoadPackageWithFlags(string(path), base.Cwd(), &stk, nil, 0))
			}
		}
	} else {
		pkglistbytes, err := buildid.ReadELFNote(shlibpath, "Go\x00\x00", 1)
		if err != nil {
			base.Fatalf("readELFNote failed: %v", err)
		}
		scanner := bufio.NewScanner(bytes.NewBuffer(pkglistbytes))
		for scanner.Scan() {
			t := scanner.Text()
			pkgs = append(pkgs, load.LoadPackageWithFlags(t, base.Cwd(), &stk, nil, 0))
		}
	}
	return
}

// cacheAction looks up {mode, p} in the cache and returns the resulting action.
// If the cache has no such action, f() is recorded and returned.
// TODO(rsc): Change the second key from *load.Package to interface{},
// to make the caching in linkShared less awkward?
func (b *Builder) cacheAction(mode string, p *load.Package, f func() *Action) *Action {
	a := b.actionCache[cacheKey{mode, p}]
	if a == nil {
		a = f()
		b.actionCache[cacheKey{mode, p}] = a
	}
	return a
}

// AutoAction returns the "right" action for go build or go install of p.
func (b *Builder) AutoAction(mode, depMode BuildMode, p *load.Package) *Action {
	if p.Name == "main" {
		return b.LinkAction(mode, depMode, p)
	}
	return b.CompileAction(mode, depMode, p)
}

// buildActor implements the Actor interface for package build
// actions. For most package builds this simply means invoking th
// *Builder.build method; in the case of "go test -cover" for
// a package with no test files, we stores some additional state
// information in the build actor to help with reporting.
type buildActor struct {
	// name of static meta-data file fragment emitted by the cover
	// tool as part of the package build action, for selected
	// "go test -cover" runs.
	covMetaFileName string
}

// newBuildActor returns a new buildActor object, setting up the
// covMetaFileName field if 'genCoverMeta' flag is set.
func newBuildActor(p *load.Package, genCoverMeta bool) *buildActor {
	ba := &buildActor{}
	if genCoverMeta {
		ba.covMetaFileName = covcmd.MetaFileForPackage(p.ImportPath)
	}
	return ba
}

func (ba *buildActor) Act(b *Builder, ctx context.Context, a *Action) error {
	return b.build(ctx, a)
}

// pgoActionID computes the action ID for a preprocess PGO action.
func (b *Builder) pgoActionID(input string) cache.ActionID {
	h := cache.NewHash("preprocess PGO profile " + input)

	fmt.Fprintf(h, "preprocess PGO profile\n")
	fmt.Fprintf(h, "preprofile %s\n", b.toolID("preprofile"))
	fmt.Fprintf(h, "input %q\n", b.fileHash(input))

	return h.Sum()
}

// pgoActor implements the Actor interface for preprocessing PGO profiles.
type pgoActor struct {
	// input is the path to the original pprof profile.
	input string
}

func (p *pgoActor) Act(b *Builder, ctx context.Context, a *Action) error {
	if b.useCache(a, b.pgoActionID(p.input), a.Target, !b.IsCmdList) || b.IsCmdList {
		return nil
	}
	defer b.flushOutput(a)

	sh := b.Shell(a)

	if err := sh.Mkdir(a.Objdir); err != nil {
		return err
	}

	if err := sh.run(".", p.input, nil, cfg.BuildToolexec, base.Tool("preprofile"), "-o", a.Target, "-i", p.input); err != nil {
		return err
	}

	// N.B. Builder.build looks for the out in a.built, regardless of
	// whether this came from cache.
	a.built = a.Target

	if !cfg.BuildN {
		// Cache the output.
		//
		// N.B. We don't use updateBuildID here, as preprocessed PGO profiles
		// do not contain a build ID. updateBuildID is typically responsible
		// for adding to the cache, thus we must do so ourselves instead.

		r, err := os.Open(a.Target)
		if err != nil {
			return fmt.Errorf("error opening target for caching: %w", err)
		}

		c := cache.Default()
		outputID, _, err := c.Put(a.actionID, r)
		r.Close()
		if err != nil {
			return fmt.Errorf("error adding target to cache: %w", err)
		}
		if cfg.BuildX {
			sh.ShowCmd("", "%s # internal", joinUnambiguously(str.StringList("cp", a.Target, c.OutputFile(outputID))))
		}
	}

	return nil
}

// CompileAction returns the action for compiling and possibly installing
// (according to mode) the given package. The resulting action is only
// for building packages (archives), never for linking executables.
// depMode is the action (build or install) to use when building dependencies.
// To turn package main into an executable, call b.Link instead.
func (b *Builder) CompileAction(mode, depMode BuildMode, p *load.Package) *Action {
	vetOnly := mode&ModeVetOnly != 0
	mode &^= ModeVetOnly

	if mode != ModeBuild && p.Target == "" {
		// No permanent target.
		mode = ModeBuild
	}
	if mode != ModeBuild && p.Name == "main" {
		// We never install the .a file for a main package.
		mode = ModeBuild
	}

	// Construct package build action.
	a := b.cacheAction("build", p, func() *Action {
		a := &Action{
			Mode:    "build",
			Package: p,
			Actor:   newBuildActor(p, p.Internal.Cover.GenMeta),
			Objdir:  b.NewObjdir(),
		}

		if p.Error == nil || !p.Error.IsImportCycle {
			for _, p1 := range p.Internal.Imports {
				a.Deps = append(a.Deps, b.CompileAction(depMode, depMode, p1))
			}
		}

		if p.Internal.PGOProfile != "" {
			pgoAction := b.cacheAction("preprocess PGO profile "+p.Internal.PGOProfile, nil, func() *Action {
				a := &Action{
					Mode:   "preprocess PGO profile",
					Actor:  &pgoActor{input: p.Internal.PGOProfile},
					Objdir: b.NewObjdir(),
				}
				a.Target = filepath.Join(a.Objdir, "pgo.preprofile")

				return a
			})
			a.Deps = append(a.Deps, pgoAction)
		}

		if p.Standard {
			switch p.ImportPath {
			case "builtin", "unsafe":
				// Fake packages - nothing to build.
				a.Mode = "built-in package"
				a.Actor = nil
				return a
			}

			// gccgo standard library is "fake" too.
			if cfg.BuildToolchainName == "gccgo" {
				// the target name is needed for cgo.
				a.Mode = "gccgo stdlib"
				a.Target = p.Target
				a.Actor = nil
				return a
			}
		}

		return a
	})

	// Find the build action; the cache entry may have been replaced
	// by the install action during (*Builder).installAction.
	buildAction := a
	switch buildAction.Mode {
	case "build", "built-in package", "gccgo stdlib":
		// ok
	case "build-install":
		buildAction = a.Deps[0]
	default:
		panic("lost build action: " + buildAction.Mode)
	}
	buildAction.needBuild = buildAction.needBuild || !vetOnly

	// Construct install action.
	if mode == ModeInstall || mode == ModeBuggyInstall {
		a = b.installAction(a, mode)
	}

	return a
}

// VetAction returns the action for running go vet on package p.
// It depends on the action for compiling p.
// If the caller may be causing p to be installed, it is up to the caller
// to make sure that the install depends on (runs after) vet.
func (b *Builder) VetAction(mode, depMode BuildMode, p *load.Package) *Action {
	a := b.vetAction(mode, depMode, p)
	a.VetxOnly = false
	return a
}

func (b *Builder) vetAction(mode, depMode BuildMode, p *load.Package) *Action {
	// Construct vet action.
	a := b.cacheAction("vet", p, func() *Action {
		a1 := b.CompileAction(mode|ModeVetOnly, depMode, p)

		// vet expects to be able to import "fmt".
		var stk load.ImportStack
		stk.Push("vet")
		p1, err := load.LoadImportWithFlags("fmt", p.Dir, p, &stk, nil, 0)
		if err != nil {
			base.Fatalf("unexpected error loading fmt package from package %s: %v", p.ImportPath, err)
		}
		stk.Pop()
		aFmt := b.CompileAction(ModeBuild, depMode, p1)

		var deps []*Action
		if a1.buggyInstall {
			// (*Builder).vet expects deps[0] to be the package
			// and deps[1] to be "fmt". If we see buggyInstall
			// here then a1 is an install of a shared library,
			// and the real package is a1.Deps[0].
			deps = []*Action{a1.Deps[0], aFmt, a1}
		} else {
			deps = []*Action{a1, aFmt}
		}
		for _, p1 := range p.Internal.Imports {
			deps = append(deps, b.vetAction(mode, depMode, p1))
		}

		a := &Action{
			Mode:       "vet",
			Package:    p,
			Deps:       deps,
			Objdir:     a1.Objdir,
			VetxOnly:   true,
			IgnoreFail: true, // it's OK if vet of dependencies "fails" (reports problems)
		}
		if a1.Actor == nil {
			// Built-in packages like unsafe.
			return a
		}
		deps[0].needVet = true
		a.Actor = ActorFunc((*Builder).vet)
		return a
	})
	return a
}

// LinkAction returns the action for linking p into an executable
// and possibly installing the result (according to mode).
// depMode is the action (build or install) to use when compiling dependencies.
func (b *Builder) LinkAction(mode, depMode BuildMode, p *load.Package) *Action {
	// Construct link action.
	a := b.cacheAction("link", p, func() *Action {
		a := &Action{
			Mode:    "link",
			Package: p,
		}

		a1 := b.CompileAction(ModeBuild, depMode, p)
		a.Actor = ActorFunc((*Builder).link)
		a.Deps = []*Action{a1}
		a.Objdir = a1.Objdir

		// An executable file. (This is the name of a temporary file.)
		// Because we run the temporary file in 'go run' and 'go test',
		// the name will show up in ps listings. If the caller has specified
		// a name, use that instead of a.out. The binary is generated
		// in an otherwise empty subdirectory named exe to avoid
		// naming conflicts. The only possible conflict is if we were
		// to create a top-level package named exe.
		name := "a.out"
		if p.Internal.ExeName != "" {
			name = p.Internal.ExeName
		} else if (cfg.Goos == "darwin" || cfg.Goos == "windows") && cfg.BuildBuildmode == "c-shared" && p.Target != "" {
			// On OS X, the linker output name gets recorded in the
			// shared library's LC_ID_DYLIB load command.
			// The code invoking the linker knows to pass only the final
			// path element. Arrange that the path element matches what
			// we'll install it as; otherwise the library is only loadable as "a.out".
			// On Windows, DLL file name is recorded in PE file
			// export section, so do like on OS X.
			_, name = filepath.Split(p.Target)
		}
		a.Target = a.Objdir + filepath.Join("exe", name) + cfg.ExeSuffix
		a.built = a.Target
		b.addTransitiveLinkDeps(a, a1, "")

		// Sequence the build of the main package (a1) strictly after the build
		// of all other dependencies that go into the link. It is likely to be after
		// them anyway, but just make sure. This is required by the build ID-based
		// shortcut in (*Builder).useCache(a1), which will call b.linkActionID(a).
		// In order for that linkActionID call to compute the right action ID, all the
		// dependencies of a (except a1) must have completed building and have
		// recorded their build IDs.
		a1.Deps = append(a1.Deps, &Action{Mode: "nop", Deps: a.Deps[1:]})
		return a
	})

	if mode == ModeInstall || mode == ModeBuggyInstall {
		a = b.installAction(a, mode)
	}

	return a
}

// installAction returns the action for installing the result of a1.
func (b *Builder) installAction(a1 *Action, mode BuildMode) *Action {
	// Because we overwrite the build action with the install action below,
	// a1 may already be an install action fetched from the "build" cache key,
	// and the caller just doesn't realize.
	if strings.HasSuffix(a1.Mode, "-install") {
		if a1.buggyInstall && mode == ModeInstall {
			//  Congratulations! The buggy install is now a proper install.
			a1.buggyInstall = false
		}
		return a1
	}

	// If there's no actual action to build a1,
	// there's nothing to install either.
	// This happens if a1 corresponds to reusing an already-built object.
	if a1.Actor == nil {
		return a1
	}

	p := a1.Package
	return b.cacheAction(a1.Mode+"-install", p, func() *Action {
		// The install deletes the temporary build result,
		// so we need all other actions, both past and future,
		// that attempt to depend on the build to depend instead
		// on the install.

		// Make a private copy of a1 (the build action),
		// no longer accessible to any other rules.
		buildAction := new(Action)
		*buildAction = *a1

		// Overwrite a1 with the install action.
		// This takes care of updating past actions that
		// point at a1 for the build action; now they will
		// point at a1 and get the install action.
		// We also leave a1 in the action cache as the result
		// for "build", so that actions not yet created that
		// try to depend on the build will instead depend
		// on the install.
		*a1 = Action{
			Mode:    buildAction.Mode + "-install",
			Actor:   ActorFunc(BuildInstallFunc),
			Package: p,
			Objdir:  buildAction.Objdir,
			Deps:    []*Action{buildAction},
			Target:  p.Target,
			built:   p.Target,

			buggyInstall: mode == ModeBuggyInstall,
		}

		b.addInstallHeaderAction(a1)
		return a1
	})
}

// addTransitiveLinkDeps adds to the link action a all packages
// that are transitive dependencies of a1.Deps.
// That is, if a is a link of package main, a1 is the compile of package main
// and a1.Deps is the actions for building packages directly imported by
// package main (what the compiler needs). The linker needs all packages
// transitively imported by the whole program; addTransitiveLinkDeps
// makes sure those are present in a.Deps.
// If shlib is non-empty, then a corresponds to the build and installation of shlib,
// so any rebuild of shlib should not be added as a dependency.
func (b *Builder) addTransitiveLinkDeps(a, a1 *Action, shlib string) {
	// Expand Deps to include all built packages, for the linker.
	// Use breadth-first search to find rebuilt-for-test packages
	// before the standard ones.
	// TODO(rsc): Eliminate the standard ones from the action graph,
	// which will require doing a little bit more rebuilding.
	workq := []*Action{a1}
	haveDep := map[string]bool{}
	if a1.Package != nil {
		haveDep[a1.Package.ImportPath] = true
	}
	for i := 0; i < len(workq); i++ {
		a1 := workq[i]
		for _, a2 := range a1.Deps {
			// TODO(rsc): Find a better discriminator than the Mode strings, once the dust settles.
			if a2.Package == nil || (a2.Mode != "build-install" && a2.Mode != "build") || haveDep[a2.Package.ImportPath] {
				continue
			}
			haveDep[a2.Package.ImportPath] = true
			a.Deps = append(a.Deps, a2)
			if a2.Mode == "build-install" {
				a2 = a2.Deps[0] // walk children of "build" action
			}
			workq = append(workq, a2)
		}
	}

	// If this is go build -linkshared, then the link depends on the shared libraries
	// in addition to the packages themselves. (The compile steps do not.)
	if cfg.BuildLinkshared {
		haveShlib := map[string]bool{shlib: true}
		for _, a1 := range a.Deps {
			p1 := a1.Package
			if p1 == nil || p1.Shlib == "" || haveShlib[filepath.Base(p1.Shlib)] {
				continue
			}
			haveShlib[filepath.Base(p1.Shlib)] = true
			// TODO(rsc): The use of ModeInstall here is suspect, but if we only do ModeBuild,
			// we'll end up building an overall library or executable that depends at runtime
			// on other libraries that are out-of-date, which is clearly not good either.
			// We call it ModeBuggyInstall to make clear that this is not right.
			a.Deps = append(a.Deps, b.linkSharedAction(ModeBuggyInstall, ModeBuggyInstall, p1.Shlib, nil))
		}
	}
}

// addInstallHeaderAction adds an install header action to a, if needed.
// The action a should be an install action as generated by either
// b.CompileAction or b.LinkAction with mode=ModeInstall,
// and so a.Deps[0] is the corresponding build action.
func (b *Builder) addInstallHeaderAction(a *Action) {
	// Install header for cgo in c-archive and c-shared modes.
	p := a.Package
	if p.UsesCgo() && (cfg.BuildBuildmode == "c-archive" || cfg.BuildBuildmode == "c-shared") {
		hdrTarget := a.Target[:len(a.Target)-len(filepath.Ext(a.Target))] + ".h"
		if cfg.BuildContext.Compiler == "gccgo" && cfg.BuildO == "" {
			// For the header file, remove the "lib"
			// added by go/build, so we generate pkg.h
			// rather than libpkg.h.
			dir, file := filepath.Split(hdrTarget)
			file = strings.TrimPrefix(file, "lib")
			hdrTarget = filepath.Join(dir, file)
		}
		ah := &Action{
			Mode:    "install header",
			Package: a.Package,
			Deps:    []*Action{a.Deps[0]},
			Actor:   ActorFunc((*Builder).installHeader),
			Objdir:  a.Deps[0].Objdir,
			Target:  hdrTarget,
		}
		a.Deps = append(a.Deps, ah)
	}
}

// buildmodeShared takes the "go build" action a1 into the building of a shared library of a1.Deps.
// That is, the input a1 represents "go build pkgs" and the result represents "go build -buildmode=shared pkgs".
func (b *Builder) buildmodeShared(mode, depMode BuildMode, args []string, pkgs []*load.Package, a1 *Action) *Action {
	name, err := libname(args, pkgs)
	if err != nil {
		base.Fatalf("%v", err)
	}
	return b.linkSharedAction(mode, depMode, name, a1)
}

// linkSharedAction takes a grouping action a1 corresponding to a list of built packages
// and returns an action that links them together into a shared library with the name shlib.
// If a1 is nil, shlib should be an absolute path to an existing shared library,
// and then linkSharedAction reads that library to find out the package list.
func (b *Builder) linkSharedAction(mode, depMode BuildMode, shlib string, a1 *Action) *Action {
	fullShlib := shlib
	shlib = filepath.Base(shlib)
	a := b.cacheAction("build-shlib "+shlib, nil, func() *Action {
		if a1 == nil {
			// TODO(rsc): Need to find some other place to store config,
			// not in pkg directory. See golang.org/issue/22196.
			pkgs := readpkglist(fullShlib)
			a1 = &Action{
				Mode: "shlib packages",
			}
			for _, p := range pkgs {
				a1.Deps = append(a1.Deps, b.CompileAction(mode, depMode, p))
			}
		}

		// Fake package to hold ldflags.
		// As usual shared libraries are a kludgy, abstraction-violating special case:
		// we let them use the flags specified for the command-line arguments.
		p := &load.Package{}
		p.Internal.CmdlinePkg = true
		p.Internal.Ldflags = load.BuildLdflags.For(p)
		p.Internal.Gccgoflags = load.BuildGccgoflags.For(p)

		// Add implicit dependencies to pkgs list.
		// Currently buildmode=shared forces external linking mode, and
		// external linking mode forces an import of runtime/cgo (and
		// math on arm). So if it was not passed on the command line and
		// it is not present in another shared library, add it here.
		// TODO(rsc): Maybe this should only happen if "runtime" is in the original package set.
		// TODO(rsc): This should probably be changed to use load.LinkerDeps(p).
		// TODO(rsc): We don't add standard library imports for gccgo
		// because they are all always linked in anyhow.
		// Maybe load.LinkerDeps should be used and updated.
		a := &Action{
			Mode:    "go build -buildmode=shared",
			Package: p,
			Objdir:  b.NewObjdir(),
			Actor:   ActorFunc((*Builder).linkShared),
			Deps:    []*Action{a1},
		}
		a.Target = filepath.Join(a.Objdir, shlib)
		if cfg.BuildToolchainName != "gccgo" {
			add := func(a1 *Action, pkg string, force bool) {
				for _, a2 := range a1.Deps {
					if a2.Package != nil && a2.Package.ImportPath == pkg {
						return
					}
				}
				var stk load.ImportStack
				p := load.LoadPackageWithFlags(pkg, base.Cwd(), &stk, nil, 0)
				if p.Error != nil {
					base.Fatalf("load %s: %v", pkg, p.Error)
				}
				// Assume that if pkg (runtime/cgo or math)
				// is already accounted for in a different shared library,
				// then that shared library also contains runtime,
				// so that anything we do will depend on that library,
				// so we don't need to include pkg in our shared library.
				if force || p.Shlib == "" || filepath.Base(p.Shlib) == pkg {
					a1.Deps = append(a1.Deps, b.CompileAction(depMode, depMode, p))
				}
			}
			add(a1, "runtime/cgo", false)
			if cfg.Goarch == "arm" {
				add(a1, "math", false)
			}

			// The linker step still needs all the usual linker deps.
			// (For example, the linker always opens runtime.a.)
			ldDeps, err := load.LinkerDeps(nil)
			if err != nil {
				base.Error(err)
			}
			for _, dep := range ldDeps {
				add(a, dep, true)
			}
		}
		b.addTransitiveLinkDeps(a, a1, shlib)
		return a
	})

	// Install result.
	if (mode == ModeInstall || mode == ModeBuggyInstall) && a.Actor != nil {
		buildAction := a

		a = b.cacheAction("install-shlib "+shlib, nil, func() *Action {
			// Determine the eventual install target.
			// The install target is root/pkg/shlib, where root is the source root
			// in which all the packages lie.
			// TODO(rsc): Perhaps this cross-root check should apply to the full
			// transitive package dependency list, not just the ones named
			// on the command line?
			pkgDir := a1.Deps[0].Package.Internal.Build.PkgTargetRoot
			for _, a2 := range a1.Deps {
				if dir := a2.Package.Internal.Build.PkgTargetRoot; dir != pkgDir {
					base.Fatalf("installing shared library: cannot use packages %s and %s from different roots %s and %s",
						a1.Deps[0].Package.ImportPath,
						a2.Package.ImportPath,
						pkgDir,
						dir)
				}
			}
			// TODO(rsc): Find out and explain here why gccgo is different.
			if cfg.BuildToolchainName == "gccgo" {
				pkgDir = filepath.Join(pkgDir, "shlibs")
			}
			target := filepath.Join(pkgDir, shlib)

			a := &Action{
				Mode:   "go install -buildmode=shared",
				Objdir: buildAction.Objdir,
				Actor:  ActorFunc(BuildInstallFunc),
				Deps:   []*Action{buildAction},
				Target: target,
			}
			for _, a2 := range buildAction.Deps[0].Deps {
				p := a2.Package
				pkgTargetRoot := p.Internal.Build.PkgTargetRoot
				if pkgTargetRoot == "" {
					continue
				}
				a.Deps = append(a.Deps, &Action{
					Mode:    "shlibname",
					Package: p,
					Actor:   ActorFunc((*Builder).installShlibname),
					Target:  filepath.Join(pkgTargetRoot, p.ImportPath+".shlibname"),
					Deps:    []*Action{a.Deps[0]},
				})
			}
			return a
		})
	}

	return a
}
