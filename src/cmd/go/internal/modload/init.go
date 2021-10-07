// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modload

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"go/build"
	"internal/lazyregexp"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/fsys"
	"cmd/go/internal/lockedfile"
	"cmd/go/internal/modconv"
	"cmd/go/internal/modfetch"
	"cmd/go/internal/search"

	"golang.org/x/mod/modfile"
	"golang.org/x/mod/module"
	"golang.org/x/mod/semver"
)

// Variables set by other packages.
//
// TODO(#40775): See if these can be plumbed as explicit parameters.
var (
	// RootMode determines whether a module root is needed.
	RootMode Root

	// ForceUseModules may be set to force modules to be enabled when
	// GO111MODULE=auto or to report an error when GO111MODULE=off.
	ForceUseModules bool

	allowMissingModuleImports bool
)

// Variables set in Init.
var (
	initialized bool
	modRoot     string
	gopath      string
)

// Variables set in initTarget (during {Load,Create}ModFile).
var (
	Target module.Version

	// targetPrefix is the path prefix for packages in Target, without a trailing
	// slash. For most modules, targetPrefix is just Target.Path, but the
	// standard-library module "std" has an empty prefix.
	targetPrefix string

	// targetInGorootSrc caches whether modRoot is within GOROOT/src.
	// The "std" module is special within GOROOT/src, but not otherwise.
	targetInGorootSrc bool
)

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
	Init()
	if modFile == nil {
		die()
	}
	return modFile
}

func BinDir() string {
	Init()
	return filepath.Join(gopath, "bin")
}

// Init determines whether module mode is enabled, locates the root of the
// current module (if any), sets environment variables for Git subprocesses, and
// configures the cfg, codehost, load, modfetch, and search packages for use
// with modules.
func Init() {
	if initialized {
		return
	}
	initialized = true

	// Keep in sync with WillBeEnabled. We perform extra validation here, and
	// there are lots of diagnostics and side effects, so we can't use
	// WillBeEnabled directly.
	var mustUseModules bool
	env := cfg.Getenv("GO111MODULE")
	switch env {
	default:
		base.Fatalf("go: unknown environment setting GO111MODULE=%s", env)
	case "auto":
		mustUseModules = ForceUseModules
	case "on", "":
		mustUseModules = true
	case "off":
		if ForceUseModules {
			base.Fatalf("go: modules disabled by GO111MODULE=off; see 'go help modules'")
		}
		mustUseModules = false
		return
	}

	if err := fsys.Init(base.Cwd()); err != nil {
		base.Fatalf("go: %v", err)
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

	// Disable any ssh connection pooling by Git.
	// If a Git subprocess forks a child into the background to cache a new connection,
	// that child keeps stdout/stderr open. After the Git subprocess exits,
	// os /exec expects to be able to read from the stdout/stderr pipe
	// until EOF to get all the data that the Git subprocess wrote before exiting.
	// The EOF doesn't come until the child exits too, because the child
	// is holding the write end of the pipe.
	// This is unfortunate, but it has come up at least twice
	// (see golang.org/issue/13453 and golang.org/issue/16104)
	// and confuses users when it does.
	// If the user has explicitly set GIT_SSH or GIT_SSH_COMMAND,
	// assume they know what they are doing and don't step on it.
	// But default to turning off ControlMaster.
	if os.Getenv("GIT_SSH") == "" && os.Getenv("GIT_SSH_COMMAND") == "" {
		os.Setenv("GIT_SSH_COMMAND", "ssh -o ControlMaster=no -o BatchMode=yes")
	}

	if os.Getenv("GCM_INTERACTIVE") == "" {
		os.Setenv("GCM_INTERACTIVE", "never")
	}

	if modRoot != "" {
		// modRoot set before Init was called ("go mod init" does this).
		// No need to search for go.mod.
	} else if RootMode == NoRoot {
		if cfg.ModFile != "" && !base.InGOFLAGS("-modfile") {
			base.Fatalf("go: -modfile cannot be used with commands that ignore the current module")
		}
		modRoot = ""
	} else {
		modRoot = findModuleRoot(base.Cwd())
		if modRoot == "" {
			if cfg.ModFile != "" {
				base.Fatalf("go: cannot find main module, but -modfile was set.\n\t-modfile cannot be used to set the module root directory.")
			}
			if RootMode == NeedRoot {
				base.Fatalf("go: %v", ErrNoModRoot)
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
			modRoot = ""
			fmt.Fprintf(os.Stderr, "go: warning: ignoring go.mod in system temp root %v\n", os.TempDir())
			if !mustUseModules {
				return
			}
		}
	}
	if cfg.ModFile != "" && !strings.HasSuffix(cfg.ModFile, ".mod") {
		base.Fatalf("go: -modfile=%s: file does not have .mod extension", cfg.ModFile)
	}

	// We're in module mode. Set any global variables that need to be set.
	cfg.ModulesEnabled = true
	setDefaultBuildMod()
	list := filepath.SplitList(cfg.BuildContext.GOPATH)
	if len(list) == 0 || list[0] == "" {
		base.Fatalf("missing $GOPATH")
	}
	gopath = list[0]
	if _, err := fsys.Stat(filepath.Join(gopath, "go.mod")); err == nil {
		base.Fatalf("$GOPATH/go.mod exists but should not")
	}

	if modRoot == "" {
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
		modfetch.GoSumFile = strings.TrimSuffix(ModFilePath(), ".mod") + ".sum"
		search.SetModRoot(modRoot)
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
	if modRoot != "" || cfg.ModulesEnabled {
		// Already enabled.
		return true
	}
	if initialized {
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

	if modRoot := findModuleRoot(base.Cwd()); modRoot == "" {
		// GO111MODULE is 'auto', and we can't find a module root.
		// Stay in GOPATH mode.
		return false
	} else if search.InDir(modRoot, os.TempDir()) == "." {
		// If you create /tmp/go.mod for experimenting,
		// then any tests that create work directories under /tmp
		// will find it and get modules when they're not expecting them.
		// It's a bit of a peculiar thing to disallow but quite mysterious
		// when it happens. See golang.org/issue/26708.
		return false
	}
	return true
}

// Enabled reports whether modules are (or must be) enabled.
// If modules are enabled but there is no main module, Enabled returns true
// and then the first use of module information will call die
// (usually through MustModRoot).
func Enabled() bool {
	Init()
	return modRoot != "" || cfg.ModulesEnabled
}

// ModRoot returns the root of the main module.
// It calls base.Fatalf if there is no main module.
func ModRoot() string {
	if !HasModRoot() {
		die()
	}
	return modRoot
}

// HasModRoot reports whether a main module is present.
// HasModRoot may return false even if Enabled returns true: for example, 'get'
// does not require a main module.
func HasModRoot() bool {
	Init()
	return modRoot != ""
}

// ModFilePath returns the effective path of the go.mod file. Normally, this
// "go.mod" in the directory returned by ModRoot, but the -modfile flag may
// change its location. ModFilePath calls base.Fatalf if there is no main
// module, even if -modfile is set.
func ModFilePath() string {
	if !HasModRoot() {
		die()
	}
	if cfg.ModFile != "" {
		return cfg.ModFile
	}
	return filepath.Join(modRoot, "go.mod")
}

func die() {
	if cfg.Getenv("GO111MODULE") == "off" {
		base.Fatalf("go: modules disabled by GO111MODULE=off; see 'go help modules'")
	}
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
	base.Fatalf("go: %v", ErrNoModRoot)
}

var ErrNoModRoot = errors.New("go.mod file not found in current directory or any parent directory; see 'go help modules'")

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

// LoadModFile sets Target and, if there is a main module, parses the initial
// build list from its go.mod file.
//
// LoadModFile may make changes in memory, like adding a go directive and
// ensuring requirements are consistent, and will write those changes back to
// disk unless DisallowWriteGoMod is in effect.
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
func LoadModFile(ctx context.Context) *Requirements {
	rs, needCommit := loadModFile(ctx)
	if needCommit {
		commitRequirements(ctx, modFileGoVersion(), rs)
	}
	return rs
}

// loadModFile is like LoadModFile, but does not implicitly commit the
// requirements back to disk after fixing inconsistencies.
//
// If needCommit is true, after the caller makes any other needed changes to the
// returned requirements they should invoke commitRequirements to fix any
// inconsistencies that may be present in the on-disk go.mod file.
func loadModFile(ctx context.Context) (rs *Requirements, needCommit bool) {
	if requirements != nil {
		return requirements, false
	}

	Init()
	if modRoot == "" {
		Target = module.Version{Path: "command-line-arguments"}
		targetPrefix = "command-line-arguments"
		goVersion := LatestGoVersion()
		rawGoVersion.Store(Target, goVersion)
		requirements = newRequirements(modDepthFromGoVersion(goVersion), nil, nil)
		return requirements, false
	}

	gomod := ModFilePath()
	var data []byte
	var err error
	if gomodActual, ok := fsys.OverlayPath(gomod); ok {
		// Don't lock go.mod if it's part of the overlay.
		// On Plan 9, locking requires chmod, and we don't want to modify any file
		// in the overlay. See #44700.
		data, err = os.ReadFile(gomodActual)
	} else {
		data, err = lockedfile.Read(gomodActual)
	}
	if err != nil {
		base.Fatalf("go: %v", err)
	}

	var fixed bool
	f, err := modfile.Parse(gomod, data, fixVersion(ctx, &fixed))
	if err != nil {
		// Errors returned by modfile.Parse begin with file:line.
		base.Fatalf("go: errors parsing go.mod:\n%s\n", err)
	}
	if f.Module == nil {
		// No module declaration. Must add module path.
		base.Fatalf("go: no module declaration in go.mod. To specify the module path:\n\tgo mod edit -module=example.com/mod")
	}

	modFile = f
	initTarget(f.Module.Mod)
	index = indexModFile(data, f, fixed)

	if err := module.CheckImportPath(f.Module.Mod.Path); err != nil {
		if pathErr, ok := err.(*module.InvalidPathError); ok {
			pathErr.Kind = "module"
		}
		base.Fatalf("go: %v", err)
	}

	setDefaultBuildMod() // possibly enable automatic vendoring
	rs = requirementsFromModFile()
	if cfg.BuildMod == "vendor" {
		readVendorList()
		checkVendorConsistency()
		rs.initVendor(vendorList)
	}
	if rs.hasRedundantRoot() {
		// If any module path appears more than once in the roots, we know that the
		// go.mod file needs to be updated even though we have not yet loaded any
		// transitive dependencies.
		rs, err = updateRoots(ctx, rs.direct, rs, nil, nil, false)
		if err != nil {
			base.Fatalf("go: %v", err)
		}
	}

	if index.goVersionV == "" {
		// TODO(#45551): Do something more principled instead of checking
		// cfg.CmdName directly here.
		if cfg.BuildMod == "mod" && cfg.CmdName != "mod graph" && cfg.CmdName != "mod why" {
			addGoStmt(LatestGoVersion())
			if go117EnableLazyLoading {
				// We need to add a 'go' version to the go.mod file, but we must assume
				// that its existing contents match something between Go 1.11 and 1.16.
				// Go 1.11 through 1.16 have eager requirements, but the latest Go
				// version uses lazy requirements instead — so we need to cnvert the
				// requirements to be lazy.
				rs, err = convertDepth(ctx, rs, lazy)
				if err != nil {
					base.Fatalf("go: %v", err)
				}
			}
		} else {
			rawGoVersion.Store(Target, modFileGoVersion())
		}
	}

	requirements = rs
	return requirements, true
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
	modRoot = base.Cwd()
	Init()
	modFilePath := ModFilePath()
	if _, err := fsys.Stat(modFilePath); err == nil {
		base.Fatalf("go: %s already exists", modFilePath)
	}

	if modPath == "" {
		var err error
		modPath, err = findModulePath(modRoot)
		if err != nil {
			base.Fatalf("go: %v", err)
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
		base.Fatalf("go: %v", err)
	}

	fmt.Fprintf(os.Stderr, "go: creating new go.mod: module %s\n", modPath)
	modFile = new(modfile.File)
	modFile.AddModuleStmt(modPath)
	initTarget(modFile.Module.Mod)
	addGoStmt(LatestGoVersion()) // Add the go directive before converted module requirements.

	convertedFrom, err := convertLegacyConfig(modPath)
	if convertedFrom != "" {
		fmt.Fprintf(os.Stderr, "go: copying requirements from %s\n", base.ShortPath(convertedFrom))
	}
	if err != nil {
		base.Fatalf("go: %v", err)
	}

	rs := requirementsFromModFile()
	rs, err = updateRoots(ctx, rs.direct, rs, nil, nil, false)
	if err != nil {
		base.Fatalf("go: %v", err)
	}
	commitRequirements(ctx, modFileGoVersion(), rs)

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
func fixVersion(ctx context.Context, fixed *bool) modfile.VersionFixer {
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

		info, err := Query(ctx, path, vers, "", nil)
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
func AllowMissingModuleImports() {
	if initialized {
		panic("AllowMissingModuleImports after Init")
	}
	allowMissingModuleImports = true
}

// initTarget sets Target and associated variables according to modFile,
func initTarget(m module.Version) {
	Target = m
	targetPrefix = m.Path

	if rel := search.InDir(base.Cwd(), cfg.GOROOTsrc); rel != "" {
		targetInGorootSrc = true
		if m.Path == "std" {
			// The "std" module in GOROOT/src is the Go standard library. Unlike other
			// modules, the packages in the "std" module have no import-path prefix.
			//
			// Modules named "std" outside of GOROOT/src do not receive this special
			// treatment, so it is possible to run 'go test .' in other GOROOTs to
			// test individual packages using a combination of the modified package
			// and the ordinary standard library.
			// (See https://golang.org/issue/30756.)
			targetPrefix = ""
		}
	}
}

// requirementsFromModFile returns the set of non-excluded requirements from
// the global modFile.
func requirementsFromModFile() *Requirements {
	roots := make([]module.Version, 0, len(modFile.Require))
	direct := map[string]bool{}
	for _, r := range modFile.Require {
		if index != nil && index.exclude[r.Mod] {
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
	module.Sort(roots)
	rs := newRequirements(modDepthFromGoVersion(modFileGoVersion()), roots, direct)
	return rs
}

// setDefaultBuildMod sets a default value for cfg.BuildMod if the -mod flag
// wasn't provided. setDefaultBuildMod may be called multiple times.
func setDefaultBuildMod() {
	if cfg.BuildModExplicit {
		// Don't override an explicit '-mod=' argument.
		return
	}

	if cfg.CmdName == "get" || strings.HasPrefix(cfg.CmdName, "mod ") {
		// 'get' and 'go mod' commands may update go.mod automatically.
		// TODO(jayconrod): should this narrower? Should 'go mod download' or
		// 'go mod graph' update go.mod by default?
		cfg.BuildMod = "mod"
		return
	}
	if modRoot == "" {
		if allowMissingModuleImports {
			cfg.BuildMod = "mod"
		} else {
			cfg.BuildMod = "readonly"
		}
		return
	}

	if fi, err := fsys.Stat(filepath.Join(modRoot, "vendor")); err == nil && fi.IsDir() {
		modGo := "unspecified"
		if index != nil && index.goVersionV != "" {
			if semver.Compare(index.goVersionV, "v1.14") >= 0 {
				// The Go version is at least 1.14, and a vendor directory exists.
				// Set -mod=vendor by default.
				cfg.BuildMod = "vendor"
				cfg.BuildModReason = "Go version in go.mod is at least 1.14 and vendor directory exists."
				return
			} else {
				modGo = index.goVersionV[1:]
			}
		}

		// Since a vendor directory exists, we should record why we didn't use it.
		// This message won't normally be shown, but it may appear with import errors.
		cfg.BuildModReason = fmt.Sprintf("Go version in go.mod is %s, so vendor directory was not used.", modGo)
	}

	cfg.BuildMod = "readonly"
}

// convertLegacyConfig imports module requirements from a legacy vendoring
// configuration file, if one is present.
func convertLegacyConfig(modPath string) (from string, err error) {
	noneSelected := func(path string) (version string) { return "none" }
	queryPackage := func(path, rev string) (module.Version, error) {
		pkgMods, modOnly, err := QueryPattern(context.Background(), path, rev, noneSelected, nil)
		if err != nil {
			return module.Version{}, err
		}
		if len(pkgMods) > 0 {
			return pkgMods[0].Mod, nil
		}
		return modOnly.Mod, nil
	}
	for _, name := range altConfigs {
		cfg := filepath.Join(modRoot, name)
		data, err := os.ReadFile(cfg)
		if err == nil {
			convert := modconv.Converters[name]
			if convert == nil {
				return "", nil
			}
			cfg = filepath.ToSlash(cfg)
			err := modconv.ConvertLegacyConfig(modFile, cfg, data, queryPackage)
			return name, err
		}
	}
	return "", nil
}

// addGoStmt adds a go directive to the go.mod file if it does not already
// include one. The 'go' version added, if any, is the latest version supported
// by this toolchain.
func addGoStmt(v string) {
	if modFile.Go != nil && modFile.Go.Version != "" {
		return
	}
	if err := modFile.AddGoStmt(v); err != nil {
		base.Fatalf("go: internal error: %v", err)
	}
	rawGoVersion.Store(Target, v)
}

// LatestGoVersion returns the latest version of the Go language supported by
// this toolchain, like "1.17".
func LatestGoVersion() string {
	tags := build.Default.ReleaseTags
	version := tags[len(tags)-1]
	if !strings.HasPrefix(version, "go") || !modfile.GoVersionRE.MatchString(version[2:]) {
		base.Fatalf("go: internal error: unrecognized default version %q", version)
	}
	return version[2:]
}

// priorGoVersion returns the Go major release immediately preceding v,
// or v itself if v is the first Go major release (1.0) or not a supported
// Go version.
func priorGoVersion(v string) string {
	vTag := "go" + v
	tags := build.Default.ReleaseTags
	for i, tag := range tags {
		if tag == vTag {
			if i == 0 {
				return v
			}

			version := tags[i-1]
			if !strings.HasPrefix(version, "go") || !modfile.GoVersionRE.MatchString(version[2:]) {
				base.Fatalf("go: internal error: unrecognized version %q", version)
			}
			return version[2:]
		}
	}
	return v
}

var altConfigs = []string{
	"Gopkg.lock",

	"GLOCKFILE",
	"Godeps/Godeps.json",
	"dependencies.tsv",
	"glide.lock",
	"vendor.conf",
	"vendor.yml",
	"vendor/manifest",
	"vendor/vendor.json",

	".git/config",
}

func findModuleRoot(dir string) (root string) {
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

	// Look for Godeps.json declaring import path.
	data, _ := os.ReadFile(filepath.Join(dir, "Godeps/Godeps.json"))
	var cfg1 struct{ ImportPath string }
	json.Unmarshal(data, &cfg1)
	if cfg1.ImportPath != "" {
		return cfg1.ImportPath, nil
	}

	// Look for vendor.json declaring import path.
	data, _ = os.ReadFile(filepath.Join(dir, "vendor/vendor.json"))
	var cfg2 struct{ RootPath string }
	json.Unmarshal(data, &cfg2)
	if cfg2.RootPath != "" {
		return cfg2.RootPath, nil
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

var allowWriteGoMod = true

// DisallowWriteGoMod causes future calls to WriteGoMod to do nothing at all.
func DisallowWriteGoMod() {
	allowWriteGoMod = false
}

// AllowWriteGoMod undoes the effect of DisallowWriteGoMod:
// future calls to WriteGoMod will update go.mod if needed.
// Note that any past calls have been discarded, so typically
// a call to AlowWriteGoMod should be followed by a call to WriteGoMod.
func AllowWriteGoMod() {
	allowWriteGoMod = true
}

// WriteGoMod writes the current build list back to go.mod.
func WriteGoMod(ctx context.Context) {
	if !allowWriteGoMod {
		panic("WriteGoMod called while disallowed")
	}
	commitRequirements(ctx, modFileGoVersion(), LoadModFile(ctx))
}

// commitRequirements writes sets the global requirements variable to rs and
// writes its contents back to the go.mod file on disk.
func commitRequirements(ctx context.Context, goVersion string, rs *Requirements) {
	requirements = rs

	if !allowWriteGoMod {
		// Some package outside of modload promised to update the go.mod file later.
		return
	}

	if modRoot == "" {
		// We aren't in a module, so we don't have anywhere to write a go.mod file.
		return
	}

	var list []*modfile.Require
	for _, m := range rs.rootModules {
		list = append(list, &modfile.Require{
			Mod:      m,
			Indirect: !rs.direct[m.Path],
		})
	}
	if goVersion != "" {
		modFile.AddGoStmt(goVersion)
	}
	if semver.Compare("v"+modFileGoVersion(), separateIndirectVersionV) < 0 {
		modFile.SetRequire(list)
	} else {
		modFile.SetRequireSeparateIndirect(list)
	}
	modFile.Cleanup()

	dirty := index.modFileIsDirty(modFile)
	if dirty && cfg.BuildMod != "mod" {
		// If we're about to fail due to -mod=readonly,
		// prefer to report a dirty go.mod over a dirty go.sum
		base.Fatalf("go: %v", errGoModDirty)
	}

	if !dirty && cfg.CmdName != "mod tidy" {
		// The go.mod file has the same semantic content that it had before
		// (but not necessarily the same exact bytes).
		// Don't write go.mod, but write go.sum in case we added or trimmed sums.
		// 'go mod init' shouldn't write go.sum, since it will be incomplete.
		if cfg.CmdName != "mod init" {
			modfetch.WriteGoSum(keepSums(ctx, loaded, rs, addBuildListZipSums))
		}
		return
	}
	gomod := ModFilePath()
	if _, ok := fsys.OverlayPath(gomod); ok {
		if dirty {
			base.Fatalf("go: updates to go.mod needed, but go.mod is part of the overlay specified with -overlay")
		}
		return
	}

	new, err := modFile.Format()
	if err != nil {
		base.Fatalf("go: %v", err)
	}
	defer func() {
		// At this point we have determined to make the go.mod file on disk equal to new.
		index = indexModFile(new, modFile, false)

		// Update go.sum after releasing the side lock and refreshing the index.
		// 'go mod init' shouldn't write go.sum, since it will be incomplete.
		if cfg.CmdName != "mod init" {
			modfetch.WriteGoSum(keepSums(ctx, loaded, rs, addBuildListZipSums))
		}
	}()

	// Make a best-effort attempt to acquire the side lock, only to exclude
	// previous versions of the 'go' command from making simultaneous edits.
	if unlock, err := modfetch.SideLock(); err == nil {
		defer unlock()
	}

	errNoChange := errors.New("no update needed")

	err = lockedfile.Transform(ModFilePath(), func(old []byte) ([]byte, error) {
		if bytes.Equal(old, new) {
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

		return new, nil
	})

	if err != nil && err != errNoChange {
		base.Fatalf("go: updating go.mod: %v", err)
	}
}

// keepSums returns the set of modules (and go.mod file entries) for which
// checksums would be needed in order to reload the same set of packages
// loaded by the most recent call to LoadPackages or ImportFromFiles,
// including any go.mod files needed to reconstruct the MVS result,
// in addition to the checksums for every module in keepMods.
func keepSums(ctx context.Context, ld *loader, rs *Requirements, which whichSums) map[module.Version]bool {
	// Every module in the full module graph contributes its requirements,
	// so in order to ensure that the build list itself is reproducible,
	// we need sums for every go.mod in the graph (regardless of whether
	// that version is selected).
	keep := make(map[module.Version]bool)

	// Add entries for modules in the build list with paths that are prefixes of
	// paths of loaded packages. We need to retain sums for all of these modules —
	// not just the modules containing the actual packages — in order to rule out
	// ambiguous import errors the next time we load the package.
	if ld != nil {
		for _, pkg := range ld.pkgs {
			// We check pkg.mod.Path here instead of pkg.inStd because the
			// pseudo-package "C" is not in std, but not provided by any module (and
			// shouldn't force loading the whole module graph).
			if pkg.testOf != nil || (pkg.mod.Path == "" && pkg.err == nil) || module.CheckImportPath(pkg.path) != nil {
				continue
			}

			if rs.depth == lazy && pkg.mod.Path != "" {
				if v, ok := rs.rootSelected(pkg.mod.Path); ok && v == pkg.mod.Version {
					// pkg was loaded from a root module, and because the main module is
					// lazy we do not check non-root modules for conflicts for packages
					// that can be found in roots. So we only need the checksums for the
					// root modules that may contain pkg, not all possible modules.
					for prefix := pkg.path; prefix != "."; prefix = path.Dir(prefix) {
						if v, ok := rs.rootSelected(prefix); ok && v != "none" {
							m := module.Version{Path: prefix, Version: v}
							keep[resolveReplacement(m)] = true
						}
					}
					continue
				}
			}

			mg, _ := rs.Graph(ctx)
			for prefix := pkg.path; prefix != "."; prefix = path.Dir(prefix) {
				if v := mg.Selected(prefix); v != "none" {
					m := module.Version{Path: prefix, Version: v}
					keep[resolveReplacement(m)] = true
				}
			}
		}
	}

	if rs.graph.Load() == nil {
		// The module graph was not loaded, possibly because the main module is lazy
		// or possibly because we haven't needed to load the graph yet.
		// Save sums for the root modules (or their replacements), but don't
		// incur the cost of loading the graph just to find and retain the sums.
		for _, m := range rs.rootModules {
			r := resolveReplacement(m)
			keep[modkey(r)] = true
			if which == addBuildListZipSums {
				keep[r] = true
			}
		}
	} else {
		mg, _ := rs.Graph(ctx)
		mg.WalkBreadthFirst(func(m module.Version) {
			if _, ok := mg.RequiredBy(m); ok {
				// The requirements from m's go.mod file are present in the module graph,
				// so they are relevant to the MVS result regardless of whether m was
				// actually selected.
				keep[modkey(resolveReplacement(m))] = true
			}
		})

		if which == addBuildListZipSums {
			for _, m := range mg.BuildList() {
				keep[resolveReplacement(m)] = true
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

// modKey returns the module.Version under which the checksum for m's go.mod
// file is stored in the go.sum file.
func modkey(m module.Version) module.Version {
	return module.Version{Path: m.Path, Version: m.Version + "/go.mod"}
}
