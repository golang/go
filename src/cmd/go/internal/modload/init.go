// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modload

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"internal/godebugs"
	"internal/lazyregexp"
	"io"
	"maps"
	"os"
	"path"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/fips140"
	"cmd/go/internal/fsys"
	"cmd/go/internal/gover"
	"cmd/go/internal/lockedfile"
	"cmd/go/internal/modfetch"
	"cmd/go/internal/search"

	"golang.org/x/mod/modfile"
	"golang.org/x/mod/module"
)

// Variables set by other packages.
//
// TODO(#40775): See if these can be plumbed as explicit parameters.
var (
	allowMissingModuleImports bool

	// ExplicitWriteGoMod prevents LoadPackages, ListModules, and other functions
	// from updating go.mod and go.sum or reporting errors when updates are
	// needed. A package should set this if it would cause go.mod to be written
	// multiple times (for example, 'go get' calls LoadPackages multiple times) or
	// if it needs some other operation to be successful before go.mod and go.sum
	// can be written (for example, 'go mod download' must download modules before
	// adding sums to go.sum). Packages that set this are responsible for calling
	// WriteGoMod explicitly.
	ExplicitWriteGoMod bool
)

// Variables set in Init.
var (
	gopath string
)

// EnterModule resets MainModules and requirements to refer to just this one module.
func EnterModule(ctx context.Context, enterModroot string) {
	LoaderState.MainModules = nil // reset MainModules
	LoaderState.requirements = nil
	LoaderState.workFilePath = "" // Force module mode
	modfetch.Reset()

	LoaderState.modRoots = []string{enterModroot}
	LoadModFile(LoaderState, ctx)
}

// EnterWorkspace enters workspace mode from module mode, applying the updated requirements to the main
// module to that module in the workspace. There should be no calls to any of the exported
// functions of the modload package running concurrently with a call to EnterWorkspace as
// EnterWorkspace will modify the global state they depend on in a non-thread-safe way.
func EnterWorkspace(ctx context.Context) (exit func(), err error) {
	// Find the identity of the main module that will be updated before we reset modload state.
	mm := LoaderState.MainModules.mustGetSingleMainModule(LoaderState)
	// Get the updated modfile we will use for that module.
	_, _, updatedmodfile, err := UpdateGoModFromReqs(LoaderState, ctx, WriteOpts{})
	if err != nil {
		return nil, err
	}

	// Reset the state to a clean state.
	oldstate := setState(LoaderState, State{})
	LoaderState.ForceUseModules = true

	// Load in workspace mode.
	InitWorkfile(LoaderState)
	LoadModFile(LoaderState, ctx)

	// Update the content of the previous main module, and recompute the requirements.
	*LoaderState.MainModules.ModFile(mm) = *updatedmodfile
	LoaderState.requirements = requirementsFromModFiles(LoaderState, ctx, LoaderState.MainModules.workFile, slices.Collect(maps.Values(LoaderState.MainModules.modFiles)), nil)

	return func() {
		setState(LoaderState, oldstate)
	}, nil
}

type MainModuleSet struct {
	// versions are the module.Version values of each of the main modules.
	// For each of them, the Path fields are ordinary module paths and the Version
	// fields are empty strings.
	// versions is clipped (len=cap).
	versions []module.Version

	// modRoot maps each module in versions to its absolute filesystem path.
	modRoot map[module.Version]string

	// pathPrefix is the path prefix for packages in the module, without a trailing
	// slash. For most modules, pathPrefix is just version.Path, but the
	// standard-library module "std" has an empty prefix.
	pathPrefix map[module.Version]string

	// inGorootSrc caches whether modRoot is within GOROOT/src.
	// The "std" module is special within GOROOT/src, but not otherwise.
	inGorootSrc map[module.Version]bool

	modFiles map[module.Version]*modfile.File

	tools map[string]bool

	modContainingCWD module.Version

	workFile *modfile.WorkFile

	workFileReplaceMap map[module.Version]module.Version
	// highest replaced version of each module path; empty string for wildcard-only replacements
	highestReplaced map[string]string

	indexMu sync.RWMutex
	indices map[module.Version]*modFileIndex
}

func (mms *MainModuleSet) PathPrefix(m module.Version) string {
	return mms.pathPrefix[m]
}

// Versions returns the module.Version values of each of the main modules.
// For each of them, the Path fields are ordinary module paths and the Version
// fields are empty strings.
// Callers should not modify the returned slice.
func (mms *MainModuleSet) Versions() []module.Version {
	if mms == nil {
		return nil
	}
	return mms.versions
}

// Tools returns the tools defined by all the main modules.
// The key is the absolute package path of the tool.
func (mms *MainModuleSet) Tools() map[string]bool {
	if mms == nil {
		return nil
	}
	return mms.tools
}

func (mms *MainModuleSet) Contains(path string) bool {
	if mms == nil {
		return false
	}
	for _, v := range mms.versions {
		if v.Path == path {
			return true
		}
	}
	return false
}

func (mms *MainModuleSet) ModRoot(m module.Version) string {
	if mms == nil {
		return ""
	}
	return mms.modRoot[m]
}

func (mms *MainModuleSet) InGorootSrc(m module.Version) bool {
	if mms == nil {
		return false
	}
	return mms.inGorootSrc[m]
}

func (mms *MainModuleSet) mustGetSingleMainModule(loaderstate *State) module.Version {
	if mms == nil || len(mms.versions) == 0 {
		panic("internal error: mustGetSingleMainModule called in context with no main modules")
	}
	if len(mms.versions) != 1 {
		if inWorkspaceMode(loaderstate) {
			panic("internal error: mustGetSingleMainModule called in workspace mode")
		} else {
			panic("internal error: multiple main modules present outside of workspace mode")
		}
	}
	return mms.versions[0]
}

func (mms *MainModuleSet) GetSingleIndexOrNil(loaderstate *State) *modFileIndex {
	if mms == nil {
		return nil
	}
	if len(mms.versions) == 0 {
		return nil
	}
	return mms.indices[mms.mustGetSingleMainModule(loaderstate)]
}

func (mms *MainModuleSet) Index(m module.Version) *modFileIndex {
	mms.indexMu.RLock()
	defer mms.indexMu.RUnlock()
	return mms.indices[m]
}

func (mms *MainModuleSet) SetIndex(m module.Version, index *modFileIndex) {
	mms.indexMu.Lock()
	defer mms.indexMu.Unlock()
	mms.indices[m] = index
}

func (mms *MainModuleSet) ModFile(m module.Version) *modfile.File {
	return mms.modFiles[m]
}

func (mms *MainModuleSet) WorkFile() *modfile.WorkFile {
	return mms.workFile
}

func (mms *MainModuleSet) Len() int {
	if mms == nil {
		return 0
	}
	return len(mms.versions)
}

// ModContainingCWD returns the main module containing the working directory,
// or module.Version{} if none of the main modules contain the working
// directory.
func (mms *MainModuleSet) ModContainingCWD() module.Version {
	return mms.modContainingCWD
}

func (mms *MainModuleSet) HighestReplaced() map[string]string {
	return mms.highestReplaced
}

// GoVersion returns the go version set on the single module, in module mode,
// or the go.work file in workspace mode.
func (mms *MainModuleSet) GoVersion(loaderstate *State) string {
	if inWorkspaceMode(loaderstate) {
		return gover.FromGoWork(mms.workFile)
	}
	if mms != nil && len(mms.versions) == 1 {
		f := mms.ModFile(mms.mustGetSingleMainModule(loaderstate))
		if f == nil {
			// Special case: we are outside a module, like 'go run x.go'.
			// Assume the local Go version.
			// TODO(#49228): Clean this up; see loadModFile.
			return gover.Local()
		}
		return gover.FromGoMod(f)
	}
	return gover.DefaultGoModVersion
}

// Godebugs returns the godebug lines set on the single module, in module mode,
// or on the go.work file in workspace mode.
// The caller must not modify the result.
func (mms *MainModuleSet) Godebugs(loaderstate *State) []*modfile.Godebug {
	if inWorkspaceMode(loaderstate) {
		if mms.workFile != nil {
			return mms.workFile.Godebug
		}
		return nil
	}
	if mms != nil && len(mms.versions) == 1 {
		f := mms.ModFile(mms.mustGetSingleMainModule(loaderstate))
		if f == nil {
			// Special case: we are outside a module, like 'go run x.go'.
			return nil
		}
		return f.Godebug
	}
	return nil
}

func (mms *MainModuleSet) WorkFileReplaceMap() map[module.Version]module.Version {
	return mms.workFileReplaceMap
}

type Root int

const (
	// AutoRoot is the default for most commands. modload.Init will look for
	// a go.mod file in the current directory or any parent. If none is found,
	// modules may be disabled (GO111MODULE=auto) or commands may run in a
	// limited module mode.
	AutoRoot Root = iota

	// NoRoot is used for commands that run in module mode and ignore any go.mod
	// file the current directory or in parent directories.
	NoRoot

	// NeedRoot is used for commands that must run in module mode and don't
	// make sense without a main module.
	NeedRoot
)

// ModFile returns the parsed go.mod file.
//
// Note that after calling LoadPackages or LoadModGraph,
// the require statements in the modfile.File are no longer
// the source of truth and will be ignored: edits made directly
// will be lost at the next call to WriteGoMod.
// To make permanent changes to the require statements
// in go.mod, edit it before loading.
func ModFile() *modfile.File {
	Init(LoaderState)
	modFile := LoaderState.MainModules.ModFile(LoaderState.MainModules.mustGetSingleMainModule(LoaderState))
	if modFile == nil {
		die(LoaderState)
	}
	return modFile
}

func BinDir(loaderstate *State) string {
	Init(loaderstate)
	if cfg.GOBIN != "" {
		return cfg.GOBIN
	}
	if gopath == "" {
		return ""
	}
	return filepath.Join(gopath, "bin")
}

// InitWorkfile initializes the workFilePath variable for commands that
// operate in workspace mode. It should not be called by other commands,
// for example 'go mod tidy', that don't operate in workspace mode.
func InitWorkfile(loaderstate *State) {
	// Initialize fsys early because we need overlay to read go.work file.
	fips140.Init()
	if err := fsys.Init(); err != nil {
		base.Fatal(err)
	}
	loaderstate.workFilePath = FindGoWork(loaderstate, base.Cwd())
}

// FindGoWork returns the name of the go.work file for this command,
// or the empty string if there isn't one.
// Most code should use Init and Enabled rather than use this directly.
// It is exported mainly for Go toolchain switching, which must process
// the go.work very early at startup.
func FindGoWork(loaderstate *State, wd string) string {
	if loaderstate.RootMode == NoRoot {
		return ""
	}

	switch gowork := cfg.Getenv("GOWORK"); gowork {
	case "off":
		return ""
	case "", "auto":
		return findWorkspaceFile(wd)
	default:
		if !filepath.IsAbs(gowork) {
			base.Fatalf("go: invalid GOWORK: not an absolute path")
		}
		return gowork
	}
}

// WorkFilePath returns the absolute path of the go.work file, or "" if not in
// workspace mode. WorkFilePath must be called after InitWorkfile.
func WorkFilePath(loaderstate *State) string {
	return loaderstate.workFilePath
}

// Reset clears all the initialized, cached state about the use of modules,
// so that we can start over.
func Reset() {
	setState(LoaderState, State{})
}

func setState(s *State, new State) State {
	oldState := State{
		initialized:     s.initialized,
		ForceUseModules: s.ForceUseModules,
		RootMode:        s.RootMode,
		modRoots:        s.modRoots,
		modulesEnabled:  cfg.ModulesEnabled,
		MainModules:     s.MainModules,
		requirements:    s.requirements,
	}
	s.initialized = new.initialized
	s.ForceUseModules = new.ForceUseModules
	s.RootMode = new.RootMode
	s.modRoots = new.modRoots
	cfg.ModulesEnabled = new.modulesEnabled
	s.MainModules = new.MainModules
	s.requirements = new.requirements
	s.workFilePath = new.workFilePath
	// The modfetch package's global state is used to compute
	// the go.sum file, so save and restore it along with the
	// modload state.
	oldState.modfetchState = modfetch.SetState(new.modfetchState)
	return oldState
}

type State struct {
	initialized bool

	// ForceUseModules may be set to force modules to be enabled when
	// GO111MODULE=auto or to report an error when GO111MODULE=off.
	ForceUseModules bool

	// RootMode determines whether a module root is needed.
	RootMode Root

	// These are primarily used to initialize the MainModules, and should
	// be eventually superseded by them but are still used in cases where
	// the module roots are required but MainModules has not been
	// initialized yet. Set to the modRoots of the main modules.
	// modRoots != nil implies len(modRoots) > 0
	modRoots       []string
	modulesEnabled bool
	MainModules    *MainModuleSet

	// requirements is the requirement graph for the main module.
	//
	// It is always non-nil if the main module's go.mod file has been
	// loaded.
	//
	// This variable should only be read from the loadModFile
	// function, and should only be written in the loadModFile and
	// commitRequirements functions.  All other functions that need or
	// produce a *Requirements should accept and/or return an explicit
	// parameter.
	requirements *Requirements

	// Set to the path to the go.work file, or "" if workspace mode is
	// disabled
	workFilePath  string
	modfetchState modfetch.State
}

func NewState() *State { return &State{} }

var LoaderState = NewState()

// Init determines whether module mode is enabled, locates the root of the
// current module (if any), sets environment variables for Git subprocesses, and
// configures the cfg, codehost, load, modfetch, and search packages for use
// with modules.
func Init(loaderstate *State) {
	if loaderstate.initialized {
		return
	}
	loaderstate.initialized = true

	fips140.Init()

	// Keep in sync with WillBeEnabled. We perform extra validation here, and
	// there are lots of diagnostics and side effects, so we can't use
	// WillBeEnabled directly.
	var mustUseModules bool
	env := cfg.Getenv("GO111MODULE")
	switch env {
	default:
		base.Fatalf("go: unknown environment setting GO111MODULE=%s", env)
	case "auto":
		mustUseModules = loaderstate.ForceUseModules
	case "on", "":
		mustUseModules = true
	case "off":
		if loaderstate.ForceUseModules {
			base.Fatalf("go: modules disabled by GO111MODULE=off; see 'go help modules'")
		}
		mustUseModules = false
		return
	}

	if err := fsys.Init(); err != nil {
		base.Fatal(err)
	}

	// Disable any prompting for passwords by Git.
	// Only has an effect for 2.3.0 or later, but avoiding
	// the prompt in earlier versions is just too hard.
	// If user has explicitly set GIT_TERMINAL_PROMPT=1, keep
	// prompting.
	// See golang.org/issue/9341 and golang.org/issue/12706.
	if os.Getenv("GIT_TERMINAL_PROMPT") == "" {
		os.Setenv("GIT_TERMINAL_PROMPT", "0")
	}

	if os.Getenv("GCM_INTERACTIVE") == "" {
		os.Setenv("GCM_INTERACTIVE", "never")
	}
	if loaderstate.modRoots != nil {
		// modRoot set before Init was called ("go mod init" does this).
		// No need to search for go.mod.
	} else if loaderstate.RootMode == NoRoot {
		if cfg.ModFile != "" && !base.InGOFLAGS("-modfile") {
			base.Fatalf("go: -modfile cannot be used with commands that ignore the current module")
		}
		loaderstate.modRoots = nil
	} else if loaderstate.workFilePath != "" {
		// We're in workspace mode, which implies module mode.
		if cfg.ModFile != "" {
			base.Fatalf("go: -modfile cannot be used in workspace mode")
		}
	} else {
		if modRoot := findModuleRoot(base.Cwd()); modRoot == "" {
			if cfg.ModFile != "" {
				base.Fatalf("go: cannot find main module, but -modfile was set.\n\t-modfile cannot be used to set the module root directory.")
			}
			if loaderstate.RootMode == NeedRoot {
				base.Fatal(ErrNoModRoot)
			}
			if !mustUseModules {
				// GO111MODULE is 'auto', and we can't find a module root.
				// Stay in GOPATH mode.
				return
			}
		} else if search.InDir(modRoot, os.TempDir()) == "." {
			// If you create /tmp/go.mod for experimenting,
			// then any tests that create work directories under /tmp
			// will find it and get modules when they're not expecting them.
			// It's a bit of a peculiar thing to disallow but quite mysterious
			// when it happens. See golang.org/issue/26708.
			fmt.Fprintf(os.Stderr, "go: warning: ignoring go.mod in system temp root %v\n", os.TempDir())
			if loaderstate.RootMode == NeedRoot {
				base.Fatal(ErrNoModRoot)
			}
			if !mustUseModules {
				return
			}
		} else {
			loaderstate.modRoots = []string{modRoot}
		}
	}
	if cfg.ModFile != "" && !strings.HasSuffix(cfg.ModFile, ".mod") {
		base.Fatalf("go: -modfile=%s: file does not have .mod extension", cfg.ModFile)
	}

	// We're in module mode. Set any global variables that need to be set.
	cfg.ModulesEnabled = true
	setDefaultBuildMod(loaderstate)
	list := filepath.SplitList(cfg.BuildContext.GOPATH)
	if len(list) > 0 && list[0] != "" {
		gopath = list[0]
		if _, err := fsys.Stat(filepath.Join(gopath, "go.mod")); err == nil {
			fmt.Fprintf(os.Stderr, "go: warning: ignoring go.mod in $GOPATH %v\n", gopath)
			if loaderstate.RootMode == NeedRoot {
				base.Fatal(ErrNoModRoot)
			}
			if !mustUseModules {
				return
			}
		}
	}
}

// WillBeEnabled checks whether modules should be enabled but does not
// initialize modules by installing hooks. If Init has already been called,
// WillBeEnabled returns the same result as Enabled.
//
// This function is needed to break a cycle. The main package needs to know
// whether modules are enabled in order to install the module or GOPATH version
// of 'go get', but Init reads the -modfile flag in 'go get', so it shouldn't
// be called until the command is installed and flags are parsed. Instead of
// calling Init and Enabled, the main package can call this function.
func WillBeEnabled() bool {
	if LoaderState.modRoots != nil || cfg.ModulesEnabled {
		// Already enabled.
		return true
	}
	if LoaderState.initialized {
		// Initialized, not enabled.
		return false
	}

	// Keep in sync with Init. Init does extra validation and prints warnings or
	// exits, so it can't call this function directly.
	env := cfg.Getenv("GO111MODULE")
	switch env {
	case "on", "":
		return true
	case "auto":
		break
	default:
		return false
	}

	return FindGoMod(base.Cwd()) != ""
}

// FindGoMod returns the name of the go.mod file for this command,
// or the empty string if there isn't one.
// Most code should use Init and Enabled rather than use this directly.
// It is exported mainly for Go toolchain switching, which must process
// the go.mod very early at startup.
func FindGoMod(wd string) string {
	modRoot := findModuleRoot(wd)
	if modRoot == "" {
		// GO111MODULE is 'auto', and we can't find a module root.
		// Stay in GOPATH mode.
		return ""
	}
	if search.InDir(modRoot, os.TempDir()) == "." {
		// If you create /tmp/go.mod for experimenting,
		// then any tests that create work directories under /tmp
		// will find it and get modules when they're not expecting them.
		// It's a bit of a peculiar thing to disallow but quite mysterious
		// when it happens. See golang.org/issue/26708.
		return ""
	}
	return filepath.Join(modRoot, "go.mod")
}

// Enabled reports whether modules are (or must be) enabled.
// If modules are enabled but there is no main module, Enabled returns true
// and then the first use of module information will call die
// (usually through MustModRoot).
func Enabled(loaderstate *State) bool {
	Init(loaderstate)
	return loaderstate.modRoots != nil || cfg.ModulesEnabled
}

func VendorDir(loaderstate *State) string {
	if inWorkspaceMode(loaderstate) {
		return filepath.Join(filepath.Dir(WorkFilePath(loaderstate)), "vendor")
	}
	// Even if -mod=vendor, we could be operating with no mod root (and thus no
	// vendor directory). As long as there are no dependencies that is expected
	// to work. See script/vendor_outside_module.txt.
	modRoot := loaderstate.MainModules.ModRoot(loaderstate.MainModules.mustGetSingleMainModule(loaderstate))
	if modRoot == "" {
		panic("vendor directory does not exist when in single module mode outside of a module")
	}
	return filepath.Join(modRoot, "vendor")
}

func inWorkspaceMode(loaderstate *State) bool {
	if !loaderstate.initialized {
		panic("inWorkspaceMode called before modload.Init called")
	}
	if !Enabled(loaderstate) {
		return false
	}
	return loaderstate.workFilePath != ""
}

// HasModRoot reports whether a main module or main modules are present.
// HasModRoot may return false even if Enabled returns true: for example, 'get'
// does not require a main module.
func HasModRoot(loaderstate *State) bool {
	Init(loaderstate)
	return loaderstate.modRoots != nil
}

// MustHaveModRoot checks that a main module or main modules are present,
// and calls base.Fatalf if there are no main modules.
func MustHaveModRoot(loaderstate *State) {
	Init(loaderstate)
	if !HasModRoot(loaderstate) {
		die(loaderstate)
	}
}

// ModFilePath returns the path that would be used for the go.mod
// file, if in module mode. ModFilePath calls base.Fatalf if there is no main
// module, even if -modfile is set.
func ModFilePath(loaderstate *State) string {
	MustHaveModRoot(loaderstate)
	return modFilePath(findModuleRoot(base.Cwd()))
}

func modFilePath(modRoot string) string {
	// TODO(matloob): This seems incompatible with workspaces
	// (unless the user's intention is to replace all workspace modules' modfiles?).
	// Should we produce an error in workspace mode if cfg.ModFile is set?
	if cfg.ModFile != "" {
		return cfg.ModFile
	}
	return filepath.Join(modRoot, "go.mod")
}

func die(loaderstate *State) {
	if cfg.Getenv("GO111MODULE") == "off" {
		base.Fatalf("go: modules disabled by GO111MODULE=off; see 'go help modules'")
	}
	if !inWorkspaceMode(loaderstate) {
		if dir, name := findAltConfig(base.Cwd()); dir != "" {
			rel, err := filepath.Rel(base.Cwd(), dir)
			if err != nil {
				rel = dir
			}
			cdCmd := ""
			if rel != "." {
				cdCmd = fmt.Sprintf("cd %s && ", rel)
			}
			base.Fatalf("go: cannot find main module, but found %s in %s\n\tto create a module there, run:\n\t%sgo mod init", name, dir, cdCmd)
		}
	}
	base.Fatal(ErrNoModRoot)
}

// noMainModulesError returns the appropriate error if there is no main module or
// main modules depending on whether the go command is in workspace mode.
type noMainModulesError struct{}

func (e noMainModulesError) Error() string {
	if inWorkspaceMode(LoaderState) {
		return "no modules were found in the current workspace; see 'go help work'"
	}
	return "go.mod file not found in current directory or any parent directory; see 'go help modules'"
}

var ErrNoModRoot noMainModulesError

type goModDirtyError struct{}

func (goModDirtyError) Error() string {
	if cfg.BuildModExplicit {
		return fmt.Sprintf("updates to go.mod needed, disabled by -mod=%v; to update it:\n\tgo mod tidy", cfg.BuildMod)
	}
	if cfg.BuildModReason != "" {
		return fmt.Sprintf("updates to go.mod needed, disabled by -mod=%s\n\t(%s)\n\tto update it:\n\tgo mod tidy", cfg.BuildMod, cfg.BuildModReason)
	}
	return "updates to go.mod needed; to update it:\n\tgo mod tidy"
}

var errGoModDirty error = goModDirtyError{}

// LoadWorkFile parses and checks the go.work file at the given path,
// and returns the absolute paths of the workspace modules' modroots.
// It does not modify the global state of the modload package.
func LoadWorkFile(path string) (workFile *modfile.WorkFile, modRoots []string, err error) {
	workDir := filepath.Dir(path)
	wf, err := ReadWorkFile(path)
	if err != nil {
		return nil, nil, err
	}
	seen := map[string]bool{}
	for _, d := range wf.Use {
		modRoot := d.Path
		if !filepath.IsAbs(modRoot) {
			modRoot = filepath.Join(workDir, modRoot)
		}

		if seen[modRoot] {
			return nil, nil, fmt.Errorf("error loading go.work:\n%s:%d: path %s appears multiple times in workspace", base.ShortPath(path), d.Syntax.Start.Line, modRoot)
		}
		seen[modRoot] = true
		modRoots = append(modRoots, modRoot)
	}

	for _, g := range wf.Godebug {
		if err := CheckGodebug("godebug", g.Key, g.Value); err != nil {
			return nil, nil, fmt.Errorf("error loading go.work:\n%s:%d: %w", base.ShortPath(path), g.Syntax.Start.Line, err)
		}
	}

	return wf, modRoots, nil
}

// ReadWorkFile reads and parses the go.work file at the given path.
func ReadWorkFile(path string) (*modfile.WorkFile, error) {
	path = base.ShortPath(path) // use short path in any errors
	workData, err := fsys.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading go.work: %w", err)
	}

	f, err := modfile.ParseWork(path, workData, nil)
	if err != nil {
		return nil, fmt.Errorf("errors parsing go.work:\n%w", err)
	}
	if f.Go != nil && gover.Compare(f.Go.Version, gover.Local()) > 0 && cfg.CmdName != "work edit" {
		base.Fatal(&gover.TooNewError{What: base.ShortPath(path), GoVersion: f.Go.Version})
	}
	return f, nil
}

// WriteWorkFile cleans and writes out the go.work file to the given path.
func WriteWorkFile(path string, wf *modfile.WorkFile) error {
	wf.SortBlocks()
	wf.Cleanup()
	out := modfile.Format(wf.Syntax)

	return os.WriteFile(path, out, 0666)
}

// UpdateWorkGoVersion updates the go line in wf to be at least goVers,
// reporting whether it changed the file.
func UpdateWorkGoVersion(wf *modfile.WorkFile, goVers string) (changed bool) {
	old := gover.FromGoWork(wf)
	if gover.Compare(old, goVers) >= 0 {
		return false
	}

	wf.AddGoStmt(goVers)

	if wf.Toolchain == nil {
		return true
	}

	// Drop the toolchain line if it is implied by the go line,
	// if its version is older than the version in the go line,
	// or if it is asking for a toolchain older than Go 1.21,
	// which will not understand the toolchain line.
	// Previously, a toolchain line set to the local toolchain
	// version was added so that future operations on the go file
	// would use the same toolchain logic for reproducibility.
	// This behavior seemed to cause user confusion without much
	// benefit so it was removed. See #65847.
	toolchain := wf.Toolchain.Name
	toolVers := gover.FromToolchain(toolchain)
	if toolchain == "go"+goVers || gover.Compare(toolVers, goVers) < 0 || gover.Compare(toolVers, gover.GoStrictVersion) < 0 {
		wf.DropToolchainStmt()
	}

	return true
}

// UpdateWorkFile updates comments on directory directives in the go.work
// file to include the associated module path.
func UpdateWorkFile(wf *modfile.WorkFile) {
	missingModulePaths := map[string]string{} // module directory listed in file -> abspath modroot

	for _, d := range wf.Use {
		if d.Path == "" {
			continue // d is marked for deletion.
		}
		modRoot := d.Path
		if d.ModulePath == "" {
			missingModulePaths[d.Path] = modRoot
		}
	}

	// Clean up and annotate directories.
	// TODO(matloob): update x/mod to actually add module paths.
	for moddir, absmodroot := range missingModulePaths {
		_, f, err := ReadModFile(filepath.Join(absmodroot, "go.mod"), nil)
		if err != nil {
			continue // Error will be reported if modules are loaded.
		}
		wf.AddUse(moddir, f.Module.Mod.Path)
	}
}

// LoadModFile sets Target and, if there is a main module, parses the initial
// build list from its go.mod file.
//
// LoadModFile may make changes in memory, like adding a go directive and
// ensuring requirements are consistent. The caller is responsible for ensuring
// those changes are written to disk by calling LoadPackages or ListModules
// (unless ExplicitWriteGoMod is set) or by calling WriteGoMod directly.
//
// As a side-effect, LoadModFile may change cfg.BuildMod to "vendor" if
// -mod wasn't set explicitly and automatic vendoring should be enabled.
//
// If LoadModFile or CreateModFile has already been called, LoadModFile returns
// the existing in-memory requirements (rather than re-reading them from disk).
//
// LoadModFile checks the roots of the module graph for consistency with each
// other, but unlike LoadModGraph does not load the full module graph or check
// it for global consistency. Most callers outside of the modload package should
// use LoadModGraph instead.
func LoadModFile(loaderstate *State, ctx context.Context) *Requirements {
	rs, err := loadModFile(loaderstate, ctx, nil)
	if err != nil {
		base.Fatal(err)
	}
	return rs
}

func loadModFile(loaderstate *State, ctx context.Context, opts *PackageOpts) (*Requirements, error) {
	if loaderstate.requirements != nil {
		return loaderstate.requirements, nil
	}

	Init(loaderstate)
	var workFile *modfile.WorkFile
	if inWorkspaceMode(loaderstate) {
		var err error
		workFile, loaderstate.modRoots, err = LoadWorkFile(loaderstate.workFilePath)
		if err != nil {
			return nil, err
		}
		for _, modRoot := range loaderstate.modRoots {
			sumFile := strings.TrimSuffix(modFilePath(modRoot), ".mod") + ".sum"
			modfetch.WorkspaceGoSumFiles = append(modfetch.WorkspaceGoSumFiles, sumFile)
		}
		modfetch.GoSumFile = loaderstate.workFilePath + ".sum"
	} else if len(loaderstate.modRoots) == 0 {
		// We're in module mode, but not inside a module.
		//
		// Commands like 'go build', 'go run', 'go list' have no go.mod file to
		// read or write. They would need to find and download the latest versions
		// of a potentially large number of modules with no way to save version
		// information. We can succeed slowly (but not reproducibly), but that's
		// not usually a good experience.
		//
		// Instead, we forbid resolving import paths to modules other than std and
		// cmd. Users may still build packages specified with .go files on the
		// command line, but they'll see an error if those files import anything
		// outside std.
		//
		// This can be overridden by calling AllowMissingModuleImports.
		// For example, 'go get' does this, since it is expected to resolve paths.
		//
		// See golang.org/issue/32027.
	} else {
		modfetch.GoSumFile = strings.TrimSuffix(modFilePath(loaderstate.modRoots[0]), ".mod") + ".sum"
	}
	if len(loaderstate.modRoots) == 0 {
		// TODO(#49228): Instead of creating a fake module with an empty modroot,
		// make MainModules.Len() == 0 mean that we're in module mode but not inside
		// any module.
		mainModule := module.Version{Path: "command-line-arguments"}
		loaderstate.MainModules = makeMainModules(loaderstate, []module.Version{mainModule}, []string{""}, []*modfile.File{nil}, []*modFileIndex{nil}, nil)
		var (
			goVersion string
			pruning   modPruning
			roots     []module.Version
			direct    = map[string]bool{"go": true}
		)
		if inWorkspaceMode(loaderstate) {
			// Since we are in a workspace, the Go version for the synthetic
			// "command-line-arguments" module must not exceed the Go version
			// for the workspace.
			goVersion = loaderstate.MainModules.GoVersion(loaderstate)
			pruning = workspace
			roots = []module.Version{
				mainModule,
				{Path: "go", Version: goVersion},
				{Path: "toolchain", Version: gover.LocalToolchain()},
			}
		} else {
			goVersion = gover.Local()
			pruning = pruningForGoVersion(goVersion)
			roots = []module.Version{
				{Path: "go", Version: goVersion},
				{Path: "toolchain", Version: gover.LocalToolchain()},
			}
		}
		rawGoVersion.Store(mainModule, goVersion)
		loaderstate.requirements = newRequirements(loaderstate, pruning, roots, direct)
		if cfg.BuildMod == "vendor" {
			// For issue 56536: Some users may have GOFLAGS=-mod=vendor set.
			// Make sure it behaves as though the fake module is vendored
			// with no dependencies.
			loaderstate.requirements.initVendor(loaderstate, nil)
		}
		return loaderstate.requirements, nil
	}

	var modFiles []*modfile.File
	var mainModules []module.Version
	var indices []*modFileIndex
	var errs []error
	for _, modroot := range loaderstate.modRoots {
		gomod := modFilePath(modroot)
		var fixed bool
		data, f, err := ReadModFile(gomod, fixVersion(loaderstate, ctx, &fixed))
		if err != nil {
			if inWorkspaceMode(loaderstate) {
				if tooNew, ok := err.(*gover.TooNewError); ok && !strings.HasPrefix(cfg.CmdName, "work ") {
					// Switching to a newer toolchain won't help - the go.work has the wrong version.
					// Report this more specific error, unless we are a command like 'go work use'
					// or 'go work sync', which will fix the problem after the caller sees the TooNewError
					// and switches to a newer toolchain.
					err = errWorkTooOld(gomod, workFile, tooNew.GoVersion)
				} else {
					err = fmt.Errorf("cannot load module %s listed in go.work file: %w",
						base.ShortPath(filepath.Dir(gomod)), base.ShortPathError(err))
				}
			}
			errs = append(errs, err)
			continue
		}
		if inWorkspaceMode(loaderstate) && !strings.HasPrefix(cfg.CmdName, "work ") {
			// Refuse to use workspace if its go version is too old.
			// Disable this check if we are a workspace command like work use or work sync,
			// which will fix the problem.
			mv := gover.FromGoMod(f)
			wv := gover.FromGoWork(workFile)
			if gover.Compare(mv, wv) > 0 && gover.Compare(mv, gover.GoStrictVersion) >= 0 {
				errs = append(errs, errWorkTooOld(gomod, workFile, mv))
				continue
			}
		}

		if !inWorkspaceMode(loaderstate) {
			ok := true
			for _, g := range f.Godebug {
				if err := CheckGodebug("godebug", g.Key, g.Value); err != nil {
					errs = append(errs, fmt.Errorf("error loading go.mod:\n%s:%d: %v", base.ShortPath(gomod), g.Syntax.Start.Line, err))
					ok = false
				}
			}
			if !ok {
				continue
			}
		}

		modFiles = append(modFiles, f)
		mainModule := f.Module.Mod
		mainModules = append(mainModules, mainModule)
		indices = append(indices, indexModFile(data, f, mainModule, fixed))

		if err := module.CheckImportPath(f.Module.Mod.Path); err != nil {
			if pathErr, ok := err.(*module.InvalidPathError); ok {
				pathErr.Kind = "module"
			}
			errs = append(errs, err)
		}
	}
	if len(errs) > 0 {
		return nil, errors.Join(errs...)
	}

	loaderstate.MainModules = makeMainModules(loaderstate, mainModules, loaderstate.modRoots, modFiles, indices, workFile)
	setDefaultBuildMod(loaderstate) // possibly enable automatic vendoring
	rs := requirementsFromModFiles(loaderstate, ctx, workFile, modFiles, opts)

	if cfg.BuildMod == "vendor" {
		readVendorList(VendorDir(loaderstate))
		versions := loaderstate.MainModules.Versions()
		indexes := make([]*modFileIndex, 0, len(versions))
		modFiles := make([]*modfile.File, 0, len(versions))
		modRoots := make([]string, 0, len(versions))
		for _, m := range versions {
			indexes = append(indexes, loaderstate.MainModules.Index(m))
			modFiles = append(modFiles, loaderstate.MainModules.ModFile(m))
			modRoots = append(modRoots, loaderstate.MainModules.ModRoot(m))
		}
		checkVendorConsistency(loaderstate, indexes, modFiles, modRoots)
		rs.initVendor(loaderstate, vendorList)
	}

	if inWorkspaceMode(loaderstate) {
		// We don't need to update the mod file so return early.
		loaderstate.requirements = rs
		return rs, nil
	}

	mainModule := loaderstate.MainModules.mustGetSingleMainModule(loaderstate)

	if rs.hasRedundantRoot(loaderstate) {
		// If any module path appears more than once in the roots, we know that the
		// go.mod file needs to be updated even though we have not yet loaded any
		// transitive dependencies.
		var err error
		rs, err = updateRoots(loaderstate, ctx, rs.direct, rs, nil, nil, false)
		if err != nil {
			return nil, err
		}
	}

	if loaderstate.MainModules.Index(mainModule).goVersion == "" && rs.pruning != workspace {
		// TODO(#45551): Do something more principled instead of checking
		// cfg.CmdName directly here.
		if cfg.BuildMod == "mod" && cfg.CmdName != "mod graph" && cfg.CmdName != "mod why" {
			// go line is missing from go.mod; add one there and add to derived requirements.
			v := gover.Local()
			if opts != nil && opts.TidyGoVersion != "" {
				v = opts.TidyGoVersion
			}
			addGoStmt(loaderstate.MainModules.ModFile(mainModule), mainModule, v)
			rs = overrideRoots(loaderstate, ctx, rs, []module.Version{{Path: "go", Version: v}})

			// We need to add a 'go' version to the go.mod file, but we must assume
			// that its existing contents match something between Go 1.11 and 1.16.
			// Go 1.11 through 1.16 do not support graph pruning, but the latest Go
			// version uses a pruned module graph — so we need to convert the
			// requirements to support pruning.
			if gover.Compare(v, gover.ExplicitIndirectVersion) >= 0 {
				var err error
				rs, err = convertPruning(loaderstate, ctx, rs, pruned)
				if err != nil {
					return nil, err
				}
			}
		} else {
			rawGoVersion.Store(mainModule, gover.DefaultGoModVersion)
		}
	}

	loaderstate.requirements = rs
	return loaderstate.requirements, nil
}

func errWorkTooOld(gomod string, wf *modfile.WorkFile, goVers string) error {
	verb := "lists"
	if wf == nil || wf.Go == nil {
		// A go.work file implicitly requires go1.18
		// even when it doesn't list any version.
		verb = "implicitly requires"
	}
	return fmt.Errorf("module %s listed in go.work file requires go >= %s, but go.work %s go %s; to update it:\n\tgo work use",
		base.ShortPath(filepath.Dir(gomod)), goVers, verb, gover.FromGoWork(wf))
}

// CheckReservedModulePath checks whether the module path is a reserved module path
// that can't be used for a user's module.
func CheckReservedModulePath(path string) error {
	if gover.IsToolchain(path) {
		return errors.New("module path is reserved")
	}

	return nil
}

// CreateModFile initializes a new module by creating a go.mod file.
//
// If modPath is empty, CreateModFile will attempt to infer the path from the
// directory location within GOPATH.
//
// If a vendoring configuration file is present, CreateModFile will attempt to
// translate it to go.mod directives. The resulting build list may not be
// exactly the same as in the legacy configuration (for example, we can't get
// packages at multiple versions from the same module).
func CreateModFile(ctx context.Context, modPath string) {
	modRoot := base.Cwd()
	LoaderState.modRoots = []string{modRoot}
	Init(LoaderState)
	modFilePath := modFilePath(modRoot)
	if _, err := fsys.Stat(modFilePath); err == nil {
		base.Fatalf("go: %s already exists", modFilePath)
	}

	if modPath == "" {
		var err error
		modPath, err = findModulePath(modRoot)
		if err != nil {
			base.Fatal(err)
		}
	} else if err := module.CheckImportPath(modPath); err != nil {
		if pathErr, ok := err.(*module.InvalidPathError); ok {
			pathErr.Kind = "module"
			// Same as build.IsLocalPath()
			if pathErr.Path == "." || pathErr.Path == ".." ||
				strings.HasPrefix(pathErr.Path, "./") || strings.HasPrefix(pathErr.Path, "../") {
				pathErr.Err = errors.New("is a local import path")
			}
		}
		base.Fatal(err)
	} else if err := CheckReservedModulePath(modPath); err != nil {
		base.Fatalf(`go: invalid module path %q: `, modPath)
	} else if _, _, ok := module.SplitPathVersion(modPath); !ok {
		if strings.HasPrefix(modPath, "gopkg.in/") {
			invalidMajorVersionMsg := fmt.Errorf("module paths beginning with gopkg.in/ must always have a major version suffix in the form of .vN:\n\tgo mod init %s", suggestGopkgIn(modPath))
			base.Fatalf(`go: invalid module path "%v": %v`, modPath, invalidMajorVersionMsg)
		}
		invalidMajorVersionMsg := fmt.Errorf("major version suffixes must be in the form of /vN and are only allowed for v2 or later:\n\tgo mod init %s", suggestModulePath(modPath))
		base.Fatalf(`go: invalid module path "%v": %v`, modPath, invalidMajorVersionMsg)
	}

	fmt.Fprintf(os.Stderr, "go: creating new go.mod: module %s\n", modPath)
	modFile := new(modfile.File)
	modFile.AddModuleStmt(modPath)
	LoaderState.MainModules = makeMainModules(LoaderState, []module.Version{modFile.Module.Mod}, []string{modRoot}, []*modfile.File{modFile}, []*modFileIndex{nil}, nil)
	addGoStmt(modFile, modFile.Module.Mod, gover.Local()) // Add the go directive before converted module requirements.

	rs := requirementsFromModFiles(LoaderState, ctx, nil, []*modfile.File{modFile}, nil)
	rs, err := updateRoots(LoaderState, ctx, rs.direct, rs, nil, nil, false)
	if err != nil {
		base.Fatal(err)
	}
	LoaderState.requirements = rs
	if err := commitRequirements(LoaderState, ctx, WriteOpts{}); err != nil {
		base.Fatal(err)
	}

	// Suggest running 'go mod tidy' unless the project is empty. Even if we
	// imported all the correct requirements above, we're probably missing
	// some sums, so the next build command in -mod=readonly will likely fail.
	//
	// We look for non-hidden .go files or subdirectories to determine whether
	// this is an existing project. Walking the tree for packages would be more
	// accurate, but could take much longer.
	empty := true
	files, _ := os.ReadDir(modRoot)
	for _, f := range files {
		name := f.Name()
		if strings.HasPrefix(name, ".") || strings.HasPrefix(name, "_") {
			continue
		}
		if strings.HasSuffix(name, ".go") || f.IsDir() {
			empty = false
			break
		}
	}
	if !empty {
		fmt.Fprintf(os.Stderr, "go: to add module requirements and sums:\n\tgo mod tidy\n")
	}
}

// fixVersion returns a modfile.VersionFixer implemented using the Query function.
//
// It resolves commit hashes and branch names to versions,
// canonicalizes versions that appeared in early vgo drafts,
// and does nothing for versions that already appear to be canonical.
//
// The VersionFixer sets 'fixed' if it ever returns a non-canonical version.
func fixVersion(loaderstate *State, ctx context.Context, fixed *bool) modfile.VersionFixer {
	return func(path, vers string) (resolved string, err error) {
		defer func() {
			if err == nil && resolved != vers {
				*fixed = true
			}
		}()

		// Special case: remove the old -gopkgin- hack.
		if strings.HasPrefix(path, "gopkg.in/") && strings.Contains(vers, "-gopkgin-") {
			vers = vers[strings.Index(vers, "-gopkgin-")+len("-gopkgin-"):]
		}

		// fixVersion is called speculatively on every
		// module, version pair from every go.mod file.
		// Avoid the query if it looks OK.
		_, pathMajor, ok := module.SplitPathVersion(path)
		if !ok {
			return "", &module.ModuleError{
				Path: path,
				Err: &module.InvalidVersionError{
					Version: vers,
					Err:     fmt.Errorf("malformed module path %q", path),
				},
			}
		}
		if vers != "" && module.CanonicalVersion(vers) == vers {
			if err := module.CheckPathMajor(vers, pathMajor); err != nil {
				return "", module.VersionError(module.Version{Path: path, Version: vers}, err)
			}
			return vers, nil
		}

		info, err := Query(loaderstate, ctx, path, vers, "", nil)
		if err != nil {
			return "", err
		}
		return info.Version, nil
	}
}

// AllowMissingModuleImports allows import paths to be resolved to modules
// when there is no module root. Normally, this is forbidden because it's slow
// and there's no way to make the result reproducible, but some commands
// like 'go get' are expected to do this.
//
// This function affects the default cfg.BuildMod when outside of a module,
// so it can only be called prior to Init.
func AllowMissingModuleImports(loaderstate *State) {
	if loaderstate.initialized {
		panic("AllowMissingModuleImports after Init")
	}
	allowMissingModuleImports = true
}

// makeMainModules creates a MainModuleSet and associated variables according to
// the given main modules.
func makeMainModules(loaderstate *State, ms []module.Version, rootDirs []string, modFiles []*modfile.File, indices []*modFileIndex, workFile *modfile.WorkFile) *MainModuleSet {
	for _, m := range ms {
		if m.Version != "" {
			panic("mainModulesCalled with module.Version with non empty Version field: " + fmt.Sprintf("%#v", m))
		}
	}
	modRootContainingCWD := findModuleRoot(base.Cwd())
	mainModules := &MainModuleSet{
		versions:        slices.Clip(ms),
		inGorootSrc:     map[module.Version]bool{},
		pathPrefix:      map[module.Version]string{},
		modRoot:         map[module.Version]string{},
		modFiles:        map[module.Version]*modfile.File{},
		indices:         map[module.Version]*modFileIndex{},
		highestReplaced: map[string]string{},
		tools:           map[string]bool{},
		workFile:        workFile,
	}
	var workFileReplaces []*modfile.Replace
	if workFile != nil {
		workFileReplaces = workFile.Replace
		mainModules.workFileReplaceMap = toReplaceMap(workFile.Replace)
	}
	mainModulePaths := make(map[string]bool)
	for _, m := range ms {
		if mainModulePaths[m.Path] {
			base.Errorf("go: module %s appears multiple times in workspace", m.Path)
		}
		mainModulePaths[m.Path] = true
	}
	replacedByWorkFile := make(map[string]bool)
	replacements := make(map[module.Version]module.Version)
	for _, r := range workFileReplaces {
		if mainModulePaths[r.Old.Path] && r.Old.Version == "" {
			base.Errorf("go: workspace module %v is replaced at all versions in the go.work file. To fix, remove the replacement from the go.work file or specify the version at which to replace the module.", r.Old.Path)
		}
		replacedByWorkFile[r.Old.Path] = true
		v, ok := mainModules.highestReplaced[r.Old.Path]
		if !ok || gover.ModCompare(r.Old.Path, r.Old.Version, v) > 0 {
			mainModules.highestReplaced[r.Old.Path] = r.Old.Version
		}
		replacements[r.Old] = r.New
	}
	for i, m := range ms {
		mainModules.pathPrefix[m] = m.Path
		mainModules.modRoot[m] = rootDirs[i]
		mainModules.modFiles[m] = modFiles[i]
		mainModules.indices[m] = indices[i]

		if mainModules.modRoot[m] == modRootContainingCWD {
			mainModules.modContainingCWD = m
		}

		if rel := search.InDir(rootDirs[i], cfg.GOROOTsrc); rel != "" {
			mainModules.inGorootSrc[m] = true
			if m.Path == "std" {
				// The "std" module in GOROOT/src is the Go standard library. Unlike other
				// modules, the packages in the "std" module have no import-path prefix.
				//
				// Modules named "std" outside of GOROOT/src do not receive this special
				// treatment, so it is possible to run 'go test .' in other GOROOTs to
				// test individual packages using a combination of the modified package
				// and the ordinary standard library.
				// (See https://golang.org/issue/30756.)
				mainModules.pathPrefix[m] = ""
			}
		}

		if modFiles[i] != nil {
			curModuleReplaces := make(map[module.Version]bool)
			for _, r := range modFiles[i].Replace {
				if replacedByWorkFile[r.Old.Path] {
					continue
				}
				var newV module.Version = r.New
				if WorkFilePath(loaderstate) != "" && newV.Version == "" && !filepath.IsAbs(newV.Path) {
					// Since we are in a workspace, we may be loading replacements from
					// multiple go.mod files. Relative paths in those replacement are
					// relative to the go.mod file, not the workspace, so the same string
					// may refer to two different paths and different strings may refer to
					// the same path. Convert them all to be absolute instead.
					//
					// (We could do this outside of a workspace too, but it would mean that
					// replacement paths in error strings needlessly differ from what's in
					// the go.mod file.)
					newV.Path = filepath.Join(rootDirs[i], newV.Path)
				}
				if prev, ok := replacements[r.Old]; ok && !curModuleReplaces[r.Old] && prev != newV {
					base.Fatalf("go: conflicting replacements for %v:\n\t%v\n\t%v\nuse \"go work edit -replace %v=[override]\" to resolve", r.Old, prev, newV, r.Old)
				}
				curModuleReplaces[r.Old] = true
				replacements[r.Old] = newV

				v, ok := mainModules.highestReplaced[r.Old.Path]
				if !ok || gover.ModCompare(r.Old.Path, r.Old.Version, v) > 0 {
					mainModules.highestReplaced[r.Old.Path] = r.Old.Version
				}
			}

			for _, t := range modFiles[i].Tool {
				if err := module.CheckImportPath(t.Path); err != nil {
					if e, ok := err.(*module.InvalidPathError); ok {
						e.Kind = "tool"
					}
					base.Fatal(err)
				}

				mainModules.tools[t.Path] = true
			}
		}
	}

	return mainModules
}

// requirementsFromModFiles returns the set of non-excluded requirements from
// the global modFile.
func requirementsFromModFiles(loaderstate *State, ctx context.Context, workFile *modfile.WorkFile, modFiles []*modfile.File, opts *PackageOpts) *Requirements {
	var roots []module.Version
	direct := map[string]bool{}
	var pruning modPruning
	if inWorkspaceMode(loaderstate) {
		pruning = workspace
		roots = make([]module.Version, len(loaderstate.MainModules.Versions()), 2+len(loaderstate.MainModules.Versions()))
		copy(roots, loaderstate.MainModules.Versions())
		goVersion := gover.FromGoWork(workFile)
		var toolchain string
		if workFile.Toolchain != nil {
			toolchain = workFile.Toolchain.Name
		}
		roots = appendGoAndToolchainRoots(roots, goVersion, toolchain, direct)
		direct = directRequirements(modFiles)
	} else {
		pruning = pruningForGoVersion(loaderstate.MainModules.GoVersion(loaderstate))
		if len(modFiles) != 1 {
			panic(fmt.Errorf("requirementsFromModFiles called with %v modfiles outside workspace mode", len(modFiles)))
		}
		modFile := modFiles[0]
		roots, direct = rootsFromModFile(loaderstate, loaderstate.MainModules.mustGetSingleMainModule(loaderstate), modFile, withToolchainRoot)
	}

	gover.ModSort(roots)
	rs := newRequirements(loaderstate, pruning, roots, direct)
	return rs
}

type addToolchainRoot bool

const (
	omitToolchainRoot addToolchainRoot = false
	withToolchainRoot                  = true
)

func directRequirements(modFiles []*modfile.File) map[string]bool {
	direct := make(map[string]bool)
	for _, modFile := range modFiles {
		for _, r := range modFile.Require {
			if !r.Indirect {
				direct[r.Mod.Path] = true
			}
		}
	}
	return direct
}

func rootsFromModFile(loaderstate *State, m module.Version, modFile *modfile.File, addToolchainRoot addToolchainRoot) (roots []module.Version, direct map[string]bool) {
	direct = make(map[string]bool)
	padding := 2 // Add padding for the toolchain and go version, added upon return.
	if !addToolchainRoot {
		padding = 1
	}
	roots = make([]module.Version, 0, padding+len(modFile.Require))
	for _, r := range modFile.Require {
		if index := loaderstate.MainModules.Index(m); index != nil && index.exclude[r.Mod] {
			if cfg.BuildMod == "mod" {
				fmt.Fprintf(os.Stderr, "go: dropping requirement on excluded version %s %s\n", r.Mod.Path, r.Mod.Version)
			} else {
				fmt.Fprintf(os.Stderr, "go: ignoring requirement on excluded version %s %s\n", r.Mod.Path, r.Mod.Version)
			}
			continue
		}

		roots = append(roots, r.Mod)
		if !r.Indirect {
			direct[r.Mod.Path] = true
		}
	}
	goVersion := gover.FromGoMod(modFile)
	var toolchain string
	if addToolchainRoot && modFile.Toolchain != nil {
		toolchain = modFile.Toolchain.Name
	}
	roots = appendGoAndToolchainRoots(roots, goVersion, toolchain, direct)
	return roots, direct
}

func appendGoAndToolchainRoots(roots []module.Version, goVersion, toolchain string, direct map[string]bool) []module.Version {
	// Add explicit go and toolchain versions, inferring as needed.
	roots = append(roots, module.Version{Path: "go", Version: goVersion})
	direct["go"] = true // Every module directly uses the language and runtime.

	if toolchain != "" {
		roots = append(roots, module.Version{Path: "toolchain", Version: toolchain})
		// Leave the toolchain as indirect: nothing in the user's module directly
		// imports a package from the toolchain, and (like an indirect dependency in
		// a module without graph pruning) we may remove the toolchain line
		// automatically if the 'go' version is changed so that it implies the exact
		// same toolchain.
	}
	return roots
}

// setDefaultBuildMod sets a default value for cfg.BuildMod if the -mod flag
// wasn't provided. setDefaultBuildMod may be called multiple times.
func setDefaultBuildMod(loaderstate *State) {
	if cfg.BuildModExplicit {
		if inWorkspaceMode(loaderstate) && cfg.BuildMod != "readonly" && cfg.BuildMod != "vendor" {
			switch cfg.CmdName {
			case "work sync", "mod graph", "mod verify", "mod why":
				// These commands run with BuildMod set to mod, but they don't take the
				// -mod flag, so we should never get here.
				panic("in workspace mode and -mod was set explicitly, but command doesn't support setting -mod")
			default:
				base.Fatalf("go: -mod may only be set to readonly or vendor when in workspace mode, but it is set to %q"+
					"\n\tRemove the -mod flag to use the default readonly value, "+
					"\n\tor set GOWORK=off to disable workspace mode.", cfg.BuildMod)
			}
		}
		// Don't override an explicit '-mod=' argument.
		return
	}

	// TODO(#40775): commands should pass in the module mode as an option
	// to modload functions instead of relying on an implicit setting
	// based on command name.
	switch cfg.CmdName {
	case "get", "mod download", "mod init", "mod tidy", "work sync":
		// These commands are intended to update go.mod and go.sum.
		cfg.BuildMod = "mod"
		return
	case "mod graph", "mod verify", "mod why":
		// These commands should not update go.mod or go.sum, but they should be
		// able to fetch modules not in go.sum and should not report errors if
		// go.mod is inconsistent. They're useful for debugging, and they need
		// to work in buggy situations.
		cfg.BuildMod = "mod"
		return
	case "mod vendor", "work vendor":
		cfg.BuildMod = "readonly"
		return
	}
	if loaderstate.modRoots == nil {
		if allowMissingModuleImports {
			cfg.BuildMod = "mod"
		} else {
			cfg.BuildMod = "readonly"
		}
		return
	}

	if len(loaderstate.modRoots) >= 1 {
		var goVersion string
		var versionSource string
		if inWorkspaceMode(loaderstate) {
			versionSource = "go.work"
			if wfg := loaderstate.MainModules.WorkFile().Go; wfg != nil {
				goVersion = wfg.Version
			}
		} else {
			versionSource = "go.mod"
			index := loaderstate.MainModules.GetSingleIndexOrNil(loaderstate)
			if index != nil {
				goVersion = index.goVersion
			}
		}
		vendorDir := ""
		if loaderstate.workFilePath != "" {
			vendorDir = filepath.Join(filepath.Dir(loaderstate.workFilePath), "vendor")
		} else {
			if len(loaderstate.modRoots) != 1 {
				panic(fmt.Errorf("outside workspace mode, but have %v modRoots", loaderstate.modRoots))
			}
			vendorDir = filepath.Join(loaderstate.modRoots[0], "vendor")
		}
		if fi, err := fsys.Stat(vendorDir); err == nil && fi.IsDir() {
			if goVersion != "" {
				if gover.Compare(goVersion, "1.14") < 0 {
					// The go version is less than 1.14. Don't set -mod=vendor by default.
					// Since a vendor directory exists, we should record why we didn't use it.
					// This message won't normally be shown, but it may appear with import errors.
					cfg.BuildModReason = fmt.Sprintf("Go version in "+versionSource+" is %s, so vendor directory was not used.", goVersion)
				} else {
					vendoredWorkspace, err := modulesTextIsForWorkspace(vendorDir)
					if err != nil {
						base.Fatalf("go: reading modules.txt for vendor directory: %v", err)
					}
					if vendoredWorkspace != (versionSource == "go.work") {
						if vendoredWorkspace {
							cfg.BuildModReason = "Outside workspace mode, but vendor directory is for a workspace."
						} else {
							cfg.BuildModReason = "In workspace mode, but vendor directory is not for a workspace"
						}
					} else {
						// The Go version is at least 1.14, a vendor directory exists, and
						// the modules.txt was generated in the same mode the command is running in.
						// Set -mod=vendor by default.
						cfg.BuildMod = "vendor"
						cfg.BuildModReason = "Go version in " + versionSource + " is at least 1.14 and vendor directory exists."
						return
					}
				}
			} else {
				cfg.BuildModReason = fmt.Sprintf("Go version in %s is unspecified, so vendor directory was not used.", versionSource)
			}
		}
	}

	cfg.BuildMod = "readonly"
}

func modulesTextIsForWorkspace(vendorDir string) (bool, error) {
	f, err := fsys.Open(filepath.Join(vendorDir, "modules.txt"))
	if errors.Is(err, os.ErrNotExist) {
		// Some vendor directories exist that don't contain modules.txt.
		// This mostly happens when converting to modules.
		// We want to preserve the behavior that mod=vendor is set (even though
		// readVendorList does nothing in that case).
		return false, nil
	}
	if err != nil {
		return false, err
	}
	defer f.Close()
	var buf [512]byte
	n, err := f.Read(buf[:])
	if err != nil && err != io.EOF {
		return false, err
	}
	line, _, _ := strings.Cut(string(buf[:n]), "\n")
	if annotations, ok := strings.CutPrefix(line, "## "); ok {
		for entry := range strings.SplitSeq(annotations, ";") {
			entry = strings.TrimSpace(entry)
			if entry == "workspace" {
				return true, nil
			}
		}
	}
	return false, nil
}

func mustHaveCompleteRequirements(loaderstate *State) bool {
	return cfg.BuildMod != "mod" && !inWorkspaceMode(loaderstate)
}

// addGoStmt adds a go directive to the go.mod file if it does not already
// include one. The 'go' version added, if any, is the latest version supported
// by this toolchain.
func addGoStmt(modFile *modfile.File, mod module.Version, v string) {
	if modFile.Go != nil && modFile.Go.Version != "" {
		return
	}
	forceGoStmt(modFile, mod, v)
}

func forceGoStmt(modFile *modfile.File, mod module.Version, v string) {
	if err := modFile.AddGoStmt(v); err != nil {
		base.Fatalf("go: internal error: %v", err)
	}
	rawGoVersion.Store(mod, v)
}

var altConfigs = []string{
	".git/config",
}

func findModuleRoot(dir string) (roots string) {
	if dir == "" {
		panic("dir not set")
	}
	dir = filepath.Clean(dir)

	// Look for enclosing go.mod.
	for {
		if fi, err := fsys.Stat(filepath.Join(dir, "go.mod")); err == nil && !fi.IsDir() {
			return dir
		}
		d := filepath.Dir(dir)
		if d == dir {
			break
		}
		dir = d
	}
	return ""
}

func findWorkspaceFile(dir string) (root string) {
	if dir == "" {
		panic("dir not set")
	}
	dir = filepath.Clean(dir)

	// Look for enclosing go.mod.
	for {
		f := filepath.Join(dir, "go.work")
		if fi, err := fsys.Stat(f); err == nil && !fi.IsDir() {
			return f
		}
		d := filepath.Dir(dir)
		if d == dir {
			break
		}
		if d == cfg.GOROOT {
			// As a special case, don't cross GOROOT to find a go.work file.
			// The standard library and commands built in go always use the vendored
			// dependencies, so avoid using a most likely irrelevant go.work file.
			return ""
		}
		dir = d
	}
	return ""
}

func findAltConfig(dir string) (root, name string) {
	if dir == "" {
		panic("dir not set")
	}
	dir = filepath.Clean(dir)
	if rel := search.InDir(dir, cfg.BuildContext.GOROOT); rel != "" {
		// Don't suggest creating a module from $GOROOT/.git/config
		// or a config file found in any parent of $GOROOT (see #34191).
		return "", ""
	}
	for {
		for _, name := range altConfigs {
			if fi, err := fsys.Stat(filepath.Join(dir, name)); err == nil && !fi.IsDir() {
				return dir, name
			}
		}
		d := filepath.Dir(dir)
		if d == dir {
			break
		}
		dir = d
	}
	return "", ""
}

func findModulePath(dir string) (string, error) {
	// TODO(bcmills): once we have located a plausible module path, we should
	// query version control (if available) to verify that it matches the major
	// version of the most recent tag.
	// See https://golang.org/issue/29433, https://golang.org/issue/27009, and
	// https://golang.org/issue/31549.

	// Cast about for import comments,
	// first in top-level directory, then in subdirectories.
	list, _ := os.ReadDir(dir)
	for _, info := range list {
		if info.Type().IsRegular() && strings.HasSuffix(info.Name(), ".go") {
			if com := findImportComment(filepath.Join(dir, info.Name())); com != "" {
				return com, nil
			}
		}
	}
	for _, info1 := range list {
		if info1.IsDir() {
			files, _ := os.ReadDir(filepath.Join(dir, info1.Name()))
			for _, info2 := range files {
				if info2.Type().IsRegular() && strings.HasSuffix(info2.Name(), ".go") {
					if com := findImportComment(filepath.Join(dir, info1.Name(), info2.Name())); com != "" {
						return path.Dir(com), nil
					}
				}
			}
		}
	}

	// Look for path in GOPATH.
	var badPathErr error
	for _, gpdir := range filepath.SplitList(cfg.BuildContext.GOPATH) {
		if gpdir == "" {
			continue
		}
		if rel := search.InDir(dir, filepath.Join(gpdir, "src")); rel != "" && rel != "." {
			path := filepath.ToSlash(rel)
			// gorelease will alert users publishing their modules to fix their paths.
			if err := module.CheckImportPath(path); err != nil {
				badPathErr = err
				break
			}
			return path, nil
		}
	}

	reason := "outside GOPATH, module path must be specified"
	if badPathErr != nil {
		// return a different error message if the module was in GOPATH, but
		// the module path determined above would be an invalid path.
		reason = fmt.Sprintf("bad module path inferred from directory in GOPATH: %v", badPathErr)
	}
	msg := `cannot determine module path for source directory %s (%s)

Example usage:
	'go mod init example.com/m' to initialize a v0 or v1 module
	'go mod init example.com/m/v2' to initialize a v2 module

Run 'go help mod init' for more information.
`
	return "", fmt.Errorf(msg, dir, reason)
}

var (
	importCommentRE = lazyregexp.New(`(?m)^package[ \t]+[^ \t\r\n/]+[ \t]+//[ \t]+import[ \t]+(\"[^"]+\")[ \t]*\r?\n`)
)

func findImportComment(file string) string {
	data, err := os.ReadFile(file)
	if err != nil {
		return ""
	}
	m := importCommentRE.FindSubmatch(data)
	if m == nil {
		return ""
	}
	path, err := strconv.Unquote(string(m[1]))
	if err != nil {
		return ""
	}
	return path
}

// WriteOpts control the behavior of WriteGoMod.
type WriteOpts struct {
	DropToolchain     bool // go get toolchain@none
	ExplicitToolchain bool // go get has set explicit toolchain version

	AddTools  []string // go get -tool example.com/m1
	DropTools []string // go get -tool example.com/m1@none

	// TODO(bcmills): Make 'go mod tidy' update the go version in the Requirements
	// instead of writing directly to the modfile.File
	TidyWroteGo bool // Go.Version field already updated by 'go mod tidy'
}

// WriteGoMod writes the current build list back to go.mod.
func WriteGoMod(ctx context.Context, opts WriteOpts) error {
	LoaderState.requirements = LoadModFile(LoaderState, ctx)
	return commitRequirements(LoaderState, ctx, opts)
}

var errNoChange = errors.New("no update needed")

// UpdateGoModFromReqs returns a modified go.mod file using the current
// requirements. It does not commit these changes to disk.
func UpdateGoModFromReqs(loaderstate *State, ctx context.Context, opts WriteOpts) (before, after []byte, modFile *modfile.File, err error) {
	if loaderstate.MainModules.Len() != 1 || loaderstate.MainModules.ModRoot(loaderstate.MainModules.Versions()[0]) == "" {
		// We aren't in a module, so we don't have anywhere to write a go.mod file.
		return nil, nil, nil, errNoChange
	}
	mainModule := loaderstate.MainModules.mustGetSingleMainModule(loaderstate)
	modFile = loaderstate.MainModules.ModFile(mainModule)
	if modFile == nil {
		// command-line-arguments has no .mod file to write.
		return nil, nil, nil, errNoChange
	}
	before, err = modFile.Format()
	if err != nil {
		return nil, nil, nil, err
	}

	var list []*modfile.Require
	toolchain := ""
	goVersion := ""
	for _, m := range loaderstate.requirements.rootModules {
		if m.Path == "go" {
			goVersion = m.Version
			continue
		}
		if m.Path == "toolchain" {
			toolchain = m.Version
			continue
		}
		list = append(list, &modfile.Require{
			Mod:      m,
			Indirect: !loaderstate.requirements.direct[m.Path],
		})
	}

	// Update go line.
	// Every MVS graph we consider should have go as a root,
	// and toolchain is either implied by the go line or explicitly a root.
	if goVersion == "" {
		base.Fatalf("go: internal error: missing go root module in WriteGoMod")
	}
	if gover.Compare(goVersion, gover.Local()) > 0 {
		// We cannot assume that we know how to update a go.mod to a newer version.
		return nil, nil, nil, &gover.TooNewError{What: "updating go.mod", GoVersion: goVersion}
	}
	wroteGo := opts.TidyWroteGo
	if !wroteGo && modFile.Go == nil || modFile.Go.Version != goVersion {
		alwaysUpdate := cfg.BuildMod == "mod" || cfg.CmdName == "mod tidy" || cfg.CmdName == "get"
		if modFile.Go == nil && goVersion == gover.DefaultGoModVersion && !alwaysUpdate {
			// The go.mod has no go line, the implied default Go version matches
			// what we've computed for the graph, and we're not in one of the
			// traditional go.mod-updating programs, so leave it alone.
		} else {
			wroteGo = true
			forceGoStmt(modFile, mainModule, goVersion)
		}
	}
	if toolchain == "" {
		toolchain = "go" + goVersion
	}

	toolVers := gover.FromToolchain(toolchain)
	if opts.DropToolchain || toolchain == "go"+goVersion || (gover.Compare(toolVers, gover.GoStrictVersion) < 0 && !opts.ExplicitToolchain) {
		// go get toolchain@none or toolchain matches go line or isn't valid; drop it.
		// TODO(#57001): 'go get' should reject explicit toolchains below GoStrictVersion.
		modFile.DropToolchainStmt()
	} else {
		modFile.AddToolchainStmt(toolchain)
	}

	for _, path := range opts.AddTools {
		modFile.AddTool(path)
	}

	for _, path := range opts.DropTools {
		modFile.DropTool(path)
	}

	// Update require blocks.
	if gover.Compare(goVersion, gover.SeparateIndirectVersion) < 0 {
		modFile.SetRequire(list)
	} else {
		modFile.SetRequireSeparateIndirect(list)
	}
	modFile.Cleanup()
	after, err = modFile.Format()
	if err != nil {
		return nil, nil, nil, err
	}
	return before, after, modFile, nil
}

// commitRequirements ensures go.mod and go.sum are up to date with the current
// requirements.
//
// In "mod" mode, commitRequirements writes changes to go.mod and go.sum.
//
// In "readonly" and "vendor" modes, commitRequirements returns an error if
// go.mod or go.sum are out of date in a semantically significant way.
//
// In workspace mode, commitRequirements only writes changes to go.work.sum.
func commitRequirements(loaderstate *State, ctx context.Context, opts WriteOpts) (err error) {
	if inWorkspaceMode(loaderstate) {
		// go.mod files aren't updated in workspace mode, but we still want to
		// update the go.work.sum file.
		return modfetch.WriteGoSum(ctx, keepSums(loaderstate, ctx, loaded, loaderstate.requirements, addBuildListZipSums), mustHaveCompleteRequirements(loaderstate))
	}
	_, updatedGoMod, modFile, err := UpdateGoModFromReqs(loaderstate, ctx, opts)
	if err != nil {
		if errors.Is(err, errNoChange) {
			return nil
		}
		return err
	}

	index := loaderstate.MainModules.GetSingleIndexOrNil(loaderstate)
	dirty := index.modFileIsDirty(modFile) || len(opts.DropTools) > 0 || len(opts.AddTools) > 0
	if dirty && cfg.BuildMod != "mod" {
		// If we're about to fail due to -mod=readonly,
		// prefer to report a dirty go.mod over a dirty go.sum
		return errGoModDirty
	}

	if !dirty && cfg.CmdName != "mod tidy" {
		// The go.mod file has the same semantic content that it had before
		// (but not necessarily the same exact bytes).
		// Don't write go.mod, but write go.sum in case we added or trimmed sums.
		// 'go mod init' shouldn't write go.sum, since it will be incomplete.
		if cfg.CmdName != "mod init" {
			if err := modfetch.WriteGoSum(ctx, keepSums(loaderstate, ctx, loaded, loaderstate.requirements, addBuildListZipSums), mustHaveCompleteRequirements(loaderstate)); err != nil {
				return err
			}
		}
		return nil
	}

	mainModule := loaderstate.MainModules.mustGetSingleMainModule(loaderstate)
	modFilePath := modFilePath(loaderstate.MainModules.ModRoot(mainModule))
	if fsys.Replaced(modFilePath) {
		if dirty {
			return errors.New("updates to go.mod needed, but go.mod is part of the overlay specified with -overlay")
		}
		return nil
	}
	defer func() {
		// At this point we have determined to make the go.mod file on disk equal to new.
		loaderstate.MainModules.SetIndex(mainModule, indexModFile(updatedGoMod, modFile, mainModule, false))

		// Update go.sum after releasing the side lock and refreshing the index.
		// 'go mod init' shouldn't write go.sum, since it will be incomplete.
		if cfg.CmdName != "mod init" {
			if err == nil {
				err = modfetch.WriteGoSum(ctx, keepSums(loaderstate, ctx, loaded, loaderstate.requirements, addBuildListZipSums), mustHaveCompleteRequirements(loaderstate))
			}
		}
	}()

	// Make a best-effort attempt to acquire the side lock, only to exclude
	// previous versions of the 'go' command from making simultaneous edits.
	if unlock, err := modfetch.SideLock(ctx); err == nil {
		defer unlock()
	}

	err = lockedfile.Transform(modFilePath, func(old []byte) ([]byte, error) {
		if bytes.Equal(old, updatedGoMod) {
			// The go.mod file is already equal to new, possibly as the result of some
			// other process.
			return nil, errNoChange
		}

		if index != nil && !bytes.Equal(old, index.data) {
			// The contents of the go.mod file have changed. In theory we could add all
			// of the new modules to the build list, recompute, and check whether any
			// module in *our* build list got bumped to a different version, but that's
			// a lot of work for marginal benefit. Instead, fail the command: if users
			// want to run concurrent commands, they need to start with a complete,
			// consistent module definition.
			return nil, fmt.Errorf("existing contents have changed since last read")
		}

		return updatedGoMod, nil
	})

	if err != nil && err != errNoChange {
		return fmt.Errorf("updating go.mod: %w", err)
	}
	return nil
}

// keepSums returns the set of modules (and go.mod file entries) for which
// checksums would be needed in order to reload the same set of packages
// loaded by the most recent call to LoadPackages or ImportFromFiles,
// including any go.mod files needed to reconstruct the MVS result
// or identify go versions,
// in addition to the checksums for every module in keepMods.
func keepSums(loaderstate *State, ctx context.Context, ld *loader, rs *Requirements, which whichSums) map[module.Version]bool {
	// Every module in the full module graph contributes its requirements,
	// so in order to ensure that the build list itself is reproducible,
	// we need sums for every go.mod in the graph (regardless of whether
	// that version is selected).
	keep := make(map[module.Version]bool)

	// Add entries for modules in the build list with paths that are prefixes of
	// paths of loaded packages. We need to retain sums for all of these modules —
	// not just the modules containing the actual packages — in order to rule out
	// ambiguous import errors the next time we load the package.
	keepModSumsForZipSums := true
	if ld == nil {
		if gover.Compare(loaderstate.MainModules.GoVersion(loaderstate), gover.TidyGoModSumVersion) < 0 && cfg.BuildMod != "mod" {
			keepModSumsForZipSums = false
		}
	} else {
		keepPkgGoModSums := true
		if gover.Compare(ld.requirements.GoVersion(loaderstate), gover.TidyGoModSumVersion) < 0 && (ld.Tidy || cfg.BuildMod != "mod") {
			keepPkgGoModSums = false
			keepModSumsForZipSums = false
		}
		for _, pkg := range ld.pkgs {
			// We check pkg.mod.Path here instead of pkg.inStd because the
			// pseudo-package "C" is not in std, but not provided by any module (and
			// shouldn't force loading the whole module graph).
			if pkg.testOf != nil || (pkg.mod.Path == "" && pkg.err == nil) || module.CheckImportPath(pkg.path) != nil {
				continue
			}

			// We need the checksum for the go.mod file for pkg.mod
			// so that we know what Go version to use to compile pkg.
			// However, we didn't do so before Go 1.21, and the bug is relatively
			// minor, so we maintain the previous (buggy) behavior in 'go mod tidy' to
			// avoid introducing unnecessary churn.
			if keepPkgGoModSums {
				r := resolveReplacement(loaderstate, pkg.mod)
				keep[modkey(r)] = true
			}

			if rs.pruning == pruned && pkg.mod.Path != "" {
				if v, ok := rs.rootSelected(loaderstate, pkg.mod.Path); ok && v == pkg.mod.Version {
					// pkg was loaded from a root module, and because the main module has
					// a pruned module graph we do not check non-root modules for
					// conflicts for packages that can be found in roots. So we only need
					// the checksums for the root modules that may contain pkg, not all
					// possible modules.
					for prefix := pkg.path; prefix != "."; prefix = path.Dir(prefix) {
						if v, ok := rs.rootSelected(loaderstate, prefix); ok && v != "none" {
							m := module.Version{Path: prefix, Version: v}
							r := resolveReplacement(loaderstate, m)
							keep[r] = true
						}
					}
					continue
				}
			}

			mg, _ := rs.Graph(loaderstate, ctx)
			for prefix := pkg.path; prefix != "."; prefix = path.Dir(prefix) {
				if v := mg.Selected(prefix); v != "none" {
					m := module.Version{Path: prefix, Version: v}
					r := resolveReplacement(loaderstate, m)
					keep[r] = true
				}
			}
		}
	}

	if rs.graph.Load() == nil {
		// We haven't needed to load the module graph so far.
		// Save sums for the root modules (or their replacements), but don't
		// incur the cost of loading the graph just to find and retain the sums.
		for _, m := range rs.rootModules {
			r := resolveReplacement(loaderstate, m)
			keep[modkey(r)] = true
			if which == addBuildListZipSums {
				keep[r] = true
			}
		}
	} else {
		mg, _ := rs.Graph(loaderstate, ctx)
		mg.WalkBreadthFirst(func(m module.Version) {
			if _, ok := mg.RequiredBy(m); ok {
				// The requirements from m's go.mod file are present in the module graph,
				// so they are relevant to the MVS result regardless of whether m was
				// actually selected.
				r := resolveReplacement(loaderstate, m)
				keep[modkey(r)] = true
			}
		})

		if which == addBuildListZipSums {
			for _, m := range mg.BuildList() {
				r := resolveReplacement(loaderstate, m)
				if keepModSumsForZipSums {
					keep[modkey(r)] = true // we need the go version from the go.mod file to do anything useful with the zipfile
				}
				keep[r] = true
			}
		}
	}

	return keep
}

type whichSums int8

const (
	loadedZipSumsOnly = whichSums(iota)
	addBuildListZipSums
)

// modkey returns the module.Version under which the checksum for m's go.mod
// file is stored in the go.sum file.
func modkey(m module.Version) module.Version {
	return module.Version{Path: m.Path, Version: m.Version + "/go.mod"}
}

func suggestModulePath(path string) string {
	var m string

	i := len(path)
	for i > 0 && ('0' <= path[i-1] && path[i-1] <= '9' || path[i-1] == '.') {
		i--
	}
	url := path[:i]
	url = strings.TrimSuffix(url, "/v")
	url = strings.TrimSuffix(url, "/")

	f := func(c rune) bool {
		return c > '9' || c < '0'
	}
	s := strings.FieldsFunc(path[i:], f)
	if len(s) > 0 {
		m = s[0]
	}
	m = strings.TrimLeft(m, "0")
	if m == "" || m == "1" {
		return url + "/v2"
	}

	return url + "/v" + m
}

func suggestGopkgIn(path string) string {
	var m string
	i := len(path)
	for i > 0 && (('0' <= path[i-1] && path[i-1] <= '9') || (path[i-1] == '.')) {
		i--
	}
	url := path[:i]
	url = strings.TrimSuffix(url, ".v")
	url = strings.TrimSuffix(url, "/v")
	url = strings.TrimSuffix(url, "/")

	f := func(c rune) bool {
		return c > '9' || c < '0'
	}
	s := strings.FieldsFunc(path, f)
	if len(s) > 0 {
		m = s[0]
	}

	m = strings.TrimLeft(m, "0")

	if m == "" {
		return url + ".v1"
	}
	return url + ".v" + m
}

func CheckGodebug(verb, k, v string) error {
	if strings.ContainsAny(k, " \t") {
		return fmt.Errorf("key contains space")
	}
	if strings.ContainsAny(v, " \t") {
		return fmt.Errorf("value contains space")
	}
	if strings.ContainsAny(k, ",") {
		return fmt.Errorf("key contains comma")
	}
	if strings.ContainsAny(v, ",") {
		return fmt.Errorf("value contains comma")
	}
	if k == "default" {
		if !strings.HasPrefix(v, "go") || !gover.IsValid(v[len("go"):]) {
			return fmt.Errorf("value for default= must be goVERSION")
		}
		if gover.Compare(v[len("go"):], gover.Local()) > 0 {
			return fmt.Errorf("default=%s too new (toolchain is go%s)", v, gover.Local())
		}
		return nil
	}
	for _, info := range godebugs.All {
		if k == info.Name {
			return nil
		}
	}
	return fmt.Errorf("unknown %s %q", verb, k)
}
