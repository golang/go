// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modload

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"go/build"
	"internal/lazyregexp"
	"io/ioutil"
	"os"
	"path"
	"path/filepath"
	"runtime/debug"
	"strconv"
	"strings"

	"cmd/go/internal/base"
	"cmd/go/internal/cache"
	"cmd/go/internal/cfg"
	"cmd/go/internal/load"
	"cmd/go/internal/lockedfile"
	"cmd/go/internal/modconv"
	"cmd/go/internal/modfetch"
	"cmd/go/internal/modfetch/codehost"
	"cmd/go/internal/mvs"
	"cmd/go/internal/search"

	"golang.org/x/mod/modfile"
	"golang.org/x/mod/module"
	"golang.org/x/mod/semver"
)

var (
	mustUseModules = false
	initialized    bool

	modRoot string
	Target  module.Version

	// targetPrefix is the path prefix for packages in Target, without a trailing
	// slash. For most modules, targetPrefix is just Target.Path, but the
	// standard-library module "std" has an empty prefix.
	targetPrefix string

	// targetInGorootSrc caches whether modRoot is within GOROOT/src.
	// The "std" module is special within GOROOT/src, but not otherwise.
	targetInGorootSrc bool

	gopath string

	CmdModInit   bool   // running 'go mod init'
	CmdModModule string // module argument for 'go mod init'

	allowMissingModuleImports bool
)

var modFile *modfile.File

// A modFileIndex is an index of data corresponding to a modFile
// at a specific point in time.
type modFileIndex struct {
	data         []byte
	dataNeedsFix bool // true if fixVersion applied a change while parsing data
	module       module.Version
	goVersion    string
	require      map[module.Version]requireMeta
	replace      map[module.Version]module.Version
	exclude      map[module.Version]bool
}

// index is the index of the go.mod file as of when it was last read or written.
var index *modFileIndex

type requireMeta struct {
	indirect bool
}

// ModFile returns the parsed go.mod file.
//
// Note that after calling ImportPaths or LoadBuildList,
// the require statements in the modfile.File are no longer
// the source of truth and will be ignored: edits made directly
// will be lost at the next call to WriteGoMod.
// To make permanent changes to the require statements
// in go.mod, edit it before calling ImportPaths or LoadBuildList.
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
	env := cfg.Getenv("GO111MODULE")
	switch env {
	default:
		base.Fatalf("go: unknown environment setting GO111MODULE=%s", env)
	case "auto", "":
		mustUseModules = false
	case "on":
		mustUseModules = true
	case "off":
		mustUseModules = false
		return
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
		os.Setenv("GIT_SSH_COMMAND", "ssh -o ControlMaster=no")
	}

	if CmdModInit {
		// Running 'go mod init': go.mod will be created in current directory.
		modRoot = base.Cwd
	} else {
		modRoot = findModuleRoot(base.Cwd)
		if modRoot == "" {
			if cfg.ModFile != "" {
				base.Fatalf("go: cannot find main module, but -modfile was set.\n\t-modfile cannot be used to set the module root directory.")
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
		}
	}
	if cfg.ModFile != "" && !strings.HasSuffix(cfg.ModFile, ".mod") {
		base.Fatalf("go: -modfile=%s: file does not have .mod extension", cfg.ModFile)
	}

	// We're in module mode. Install the hooks to make it work.

	if c := cache.Default(); c == nil {
		// With modules, there are no install locations for packages
		// other than the build cache.
		base.Fatalf("go: cannot use modules with build cache disabled")
	}

	list := filepath.SplitList(cfg.BuildContext.GOPATH)
	if len(list) == 0 || list[0] == "" {
		base.Fatalf("missing $GOPATH")
	}
	gopath = list[0]
	if _, err := os.Stat(filepath.Join(gopath, "go.mod")); err == nil {
		base.Fatalf("$GOPATH/go.mod exists but should not")
	}

	oldSrcMod := filepath.Join(list[0], "src/mod")
	pkgMod := filepath.Join(list[0], "pkg/mod")
	infoOld, errOld := os.Stat(oldSrcMod)
	_, errMod := os.Stat(pkgMod)
	if errOld == nil && infoOld.IsDir() && errMod != nil && os.IsNotExist(errMod) {
		os.Rename(oldSrcMod, pkgMod)
	}

	modfetch.PkgMod = pkgMod
	codehost.WorkRoot = filepath.Join(pkgMod, "cache/vcs")

	cfg.ModulesEnabled = true
	load.ModBinDir = BinDir
	load.ModLookup = Lookup
	load.ModPackageModuleInfo = PackageModuleInfo
	load.ModImportPaths = ImportPaths
	load.ModPackageBuildInfo = PackageBuildInfo
	load.ModInfoProg = ModInfoProg
	load.ModImportFromFiles = ImportFromFiles
	load.ModDirImportPath = DirImportPath

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

func init() {
	load.ModInit = Init

	// Set modfetch.PkgMod and codehost.WorkRoot unconditionally,
	// so that go clean -modcache and go mod download can run even without modules enabled.
	if list := filepath.SplitList(cfg.BuildContext.GOPATH); len(list) > 0 && list[0] != "" {
		modfetch.PkgMod = filepath.Join(list[0], "pkg/mod")
		codehost.WorkRoot = filepath.Join(list[0], "pkg/mod/cache/vcs")
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
	if modRoot != "" || mustUseModules {
		return true
	}
	if initialized {
		return false
	}

	// Keep in sync with Init. Init does extra validation and prints warnings or
	// exits, so it can't call this function directly.
	env := cfg.Getenv("GO111MODULE")
	switch env {
	case "on":
		return true
	case "auto", "":
		break
	default:
		return false
	}

	if CmdModInit {
		// Running 'go mod init': go.mod will be created in current directory.
		return true
	}
	if modRoot := findModuleRoot(base.Cwd); modRoot == "" {
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
	return modRoot != "" || mustUseModules
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

// printStackInDie causes die to print a stack trace.
//
// It is enabled by the testgo tag, and helps to diagnose paths that
// unexpectedly require a main module.
var printStackInDie = false

func die() {
	if printStackInDie {
		debug.PrintStack()
	}
	if cfg.Getenv("GO111MODULE") == "off" {
		base.Fatalf("go: modules disabled by GO111MODULE=off; see 'go help modules'")
	}
	if dir, name := findAltConfig(base.Cwd); dir != "" {
		rel, err := filepath.Rel(base.Cwd, dir)
		if err != nil {
			rel = dir
		}
		cdCmd := ""
		if rel != "." {
			cdCmd = fmt.Sprintf("cd %s && ", rel)
		}
		base.Fatalf("go: cannot find main module, but found %s in %s\n\tto create a module there, run:\n\t%sgo mod init", name, dir, cdCmd)
	}
	base.Fatalf("go: cannot find main module; see 'go help modules'")
}

// InitMod sets Target and, if there is a main module, parses the initial build
// list from its go.mod file, creating and populating that file if needed.
//
// As a side-effect, InitMod sets a default for cfg.BuildMod if it does not
// already have an explicit value.
func InitMod() {
	if len(buildList) > 0 {
		return
	}

	Init()
	if modRoot == "" {
		Target = module.Version{Path: "command-line-arguments"}
		targetPrefix = "command-line-arguments"
		buildList = []module.Version{Target}
		return
	}

	if CmdModInit {
		// Running go mod init: do legacy module conversion
		legacyModInit()
		modFileToBuildList()
		WriteGoMod()
		return
	}

	gomod := ModFilePath()
	data, err := lockedfile.Read(gomod)
	if err != nil {
		base.Fatalf("go: %v", err)
	}

	var fixed bool
	f, err := modfile.Parse(gomod, data, fixVersion(&fixed))
	if err != nil {
		// Errors returned by modfile.Parse begin with file:line.
		base.Fatalf("go: errors parsing go.mod:\n%s\n", err)
	}
	modFile = f
	index = indexModFile(data, f, fixed)

	if len(f.Syntax.Stmt) == 0 || f.Module == nil {
		// Empty mod file. Must add module path.
		path, err := findModulePath(modRoot)
		if err != nil {
			base.Fatalf("go: %v", err)
		}
		f.AddModuleStmt(path)
	}

	if len(f.Syntax.Stmt) == 1 && f.Module != nil {
		// Entire file is just a module statement.
		// Populate require if possible.
		legacyModInit()
	}

	modFileToBuildList()
	setDefaultBuildMod()
	if cfg.BuildMod == "vendor" {
		readVendorList()
		checkVendorConsistency()
	} else {
		// TODO(golang.org/issue/33326): if cfg.BuildMod != "readonly"?
		WriteGoMod()
	}
}

// fixVersion returns a modfile.VersionFixer implemented using the Query function.
//
// It resolves commit hashes and branch names to versions,
// canonicalizes versions that appeared in early vgo drafts,
// and does nothing for versions that already appear to be canonical.
//
// The VersionFixer sets 'fixed' if it ever returns a non-canonical version.
func fixVersion(fixed *bool) modfile.VersionFixer {
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
			if err := module.CheckPathMajor(vers, pathMajor); err == nil {
				return vers, nil
			}
		}

		info, err := Query(path, vers, "", nil)
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
func AllowMissingModuleImports() {
	allowMissingModuleImports = true
}

// modFileToBuildList initializes buildList from the modFile.
func modFileToBuildList() {
	Target = modFile.Module.Mod
	targetPrefix = Target.Path
	if rel := search.InDir(base.Cwd, cfg.GOROOTsrc); rel != "" {
		targetInGorootSrc = true
		if Target.Path == "std" {
			targetPrefix = ""
		}
	}

	list := []module.Version{Target}
	for _, r := range modFile.Require {
		list = append(list, r.Mod)
	}
	buildList = list
}

// setDefaultBuildMod sets a default value for cfg.BuildMod
// if it is currently empty.
func setDefaultBuildMod() {
	if cfg.BuildMod != "" {
		// Don't override an explicit '-mod=' argument.
		return
	}
	cfg.BuildMod = "mod"
	if cfg.CmdName == "get" || strings.HasPrefix(cfg.CmdName, "mod ") {
		// Don't set -mod implicitly for commands whose purpose is to
		// manipulate the build list.
		return
	}
	if modRoot == "" {
		return
	}

	if fi, err := os.Stat(filepath.Join(modRoot, "vendor")); err == nil && fi.IsDir() {
		modGo := "unspecified"
		if index.goVersion != "" {
			if semver.Compare("v"+index.goVersion, "v1.14") >= 0 {
				// The Go version is at least 1.14, and a vendor directory exists.
				// Set -mod=vendor by default.
				cfg.BuildMod = "vendor"
				cfg.BuildModReason = "Go version in go.mod is at least 1.14 and vendor directory exists."
				return
			} else {
				modGo = index.goVersion
			}
		}

		// Since a vendor directory exists, we have a non-trivial reason for
		// choosing -mod=mod, although it probably won't be used for anything.
		// Record the reason anyway for consistency.
		// It may be overridden if we switch to mod=readonly below.
		cfg.BuildModReason = fmt.Sprintf("Go version in go.mod is %s.", modGo)
	}

	p := ModFilePath()
	if fi, err := os.Stat(p); err == nil && !hasWritePerm(p, fi) {
		cfg.BuildMod = "readonly"
		cfg.BuildModReason = "go.mod file is read-only."
	}
}

// checkVendorConsistency verifies that the vendor/modules.txt file matches (if
// go 1.14) or at least does not contradict (go 1.13 or earlier) the
// requirements and replacements listed in the main module's go.mod file.
func checkVendorConsistency() {
	readVendorList()

	pre114 := false
	if modFile.Go == nil || semver.Compare("v"+modFile.Go.Version, "v1.14") < 0 {
		// Go versions before 1.14 did not include enough information in
		// vendor/modules.txt to check for consistency.
		// If we know that we're on an earlier version, relax the consistency check.
		pre114 = true
	}

	vendErrors := new(strings.Builder)
	vendErrorf := func(mod module.Version, format string, args ...interface{}) {
		detail := fmt.Sprintf(format, args...)
		if mod.Version == "" {
			fmt.Fprintf(vendErrors, "\n\t%s: %s", mod.Path, detail)
		} else {
			fmt.Fprintf(vendErrors, "\n\t%s@%s: %s", mod.Path, mod.Version, detail)
		}
	}

	for _, r := range modFile.Require {
		if !vendorMeta[r.Mod].Explicit {
			if pre114 {
				// Before 1.14, modules.txt did not indicate whether modules were listed
				// explicitly in the main module's go.mod file.
				// However, we can at least detect a version mismatch if packages were
				// vendored from a non-matching version.
				if vv, ok := vendorVersion[r.Mod.Path]; ok && vv != r.Mod.Version {
					vendErrorf(r.Mod, fmt.Sprintf("is explicitly required in go.mod, but vendor/modules.txt indicates %s@%s", r.Mod.Path, vv))
				}
			} else {
				vendErrorf(r.Mod, "is explicitly required in go.mod, but not marked as explicit in vendor/modules.txt")
			}
		}
	}

	describe := func(m module.Version) string {
		if m.Version == "" {
			return m.Path
		}
		return m.Path + "@" + m.Version
	}

	// We need to verify *all* replacements that occur in modfile: even if they
	// don't directly apply to any module in the vendor list, the replacement
	// go.mod file can affect the selected versions of other (transitive)
	// dependencies
	for _, r := range modFile.Replace {
		vr := vendorMeta[r.Old].Replacement
		if vr == (module.Version{}) {
			if pre114 && (r.Old.Version == "" || vendorVersion[r.Old.Path] != r.Old.Version) {
				// Before 1.14, modules.txt omitted wildcard replacements and
				// replacements for modules that did not have any packages to vendor.
			} else {
				vendErrorf(r.Old, "is replaced in go.mod, but not marked as replaced in vendor/modules.txt")
			}
		} else if vr != r.New {
			vendErrorf(r.Old, "is replaced by %s in go.mod, but marked as replaced by %s in vendor/modules.txt", describe(r.New), describe(vr))
		}
	}

	for _, mod := range vendorList {
		meta := vendorMeta[mod]
		if meta.Explicit {
			if _, inGoMod := index.require[mod]; !inGoMod {
				vendErrorf(mod, "is marked as explicit in vendor/modules.txt, but not explicitly required in go.mod")
			}
		}
	}

	for _, mod := range vendorReplaced {
		r := Replacement(mod)
		if r == (module.Version{}) {
			vendErrorf(mod, "is marked as replaced in vendor/modules.txt, but not replaced in go.mod")
			continue
		}
		if meta := vendorMeta[mod]; r != meta.Replacement {
			vendErrorf(mod, "is marked as replaced by %s in vendor/modules.txt, but replaced by %s in go.mod", describe(meta.Replacement), describe(r))
		}
	}

	if vendErrors.Len() > 0 {
		base.Fatalf("go: inconsistent vendoring in %s:%s\n\nrun 'go mod vendor' to sync, or use -mod=mod or -mod=readonly to ignore the vendor directory", modRoot, vendErrors)
	}
}

// Allowed reports whether module m is allowed (not excluded) by the main module's go.mod.
func Allowed(m module.Version) bool {
	return index == nil || !index.exclude[m]
}

func legacyModInit() {
	if modFile == nil {
		path, err := findModulePath(modRoot)
		if err != nil {
			base.Fatalf("go: %v", err)
		}
		fmt.Fprintf(os.Stderr, "go: creating new go.mod: module %s\n", path)
		modFile = new(modfile.File)
		modFile.AddModuleStmt(path)
		addGoStmt() // Add the go directive before converted module requirements.
	}

	for _, name := range altConfigs {
		cfg := filepath.Join(modRoot, name)
		data, err := ioutil.ReadFile(cfg)
		if err == nil {
			convert := modconv.Converters[name]
			if convert == nil {
				return
			}
			fmt.Fprintf(os.Stderr, "go: copying requirements from %s\n", base.ShortPath(cfg))
			cfg = filepath.ToSlash(cfg)
			if err := modconv.ConvertLegacyConfig(modFile, cfg, data); err != nil {
				base.Fatalf("go: %v", err)
			}
			if len(modFile.Syntax.Stmt) == 1 {
				// Add comment to avoid re-converting every time it runs.
				modFile.AddComment("// go: no requirements found in " + name)
			}
			return
		}
	}
}

// addGoStmt adds a go directive to the go.mod file if it does not already include one.
// The 'go' version added, if any, is the latest version supported by this toolchain.
func addGoStmt() {
	if modFile.Go != nil && modFile.Go.Version != "" {
		return
	}
	tags := build.Default.ReleaseTags
	version := tags[len(tags)-1]
	if !strings.HasPrefix(version, "go") || !modfile.GoVersionRE.MatchString(version[2:]) {
		base.Fatalf("go: unrecognized default version %q", version)
	}
	if err := modFile.AddGoStmt(version[2:]); err != nil {
		base.Fatalf("go: internal error: %v", err)
	}
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
		if fi, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil && !fi.IsDir() {
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
	for {
		for _, name := range altConfigs {
			if fi, err := os.Stat(filepath.Join(dir, name)); err == nil && !fi.IsDir() {
				if rel := search.InDir(dir, cfg.BuildContext.GOROOT); rel == "." {
					// Don't suggest creating a module from $GOROOT/.git/config.
					return "", ""
				}
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
	if CmdModModule != "" {
		// Running go mod init x/y/z; return x/y/z.
		if err := module.CheckImportPath(CmdModModule); err != nil {
			return "", err
		}
		return CmdModModule, nil
	}

	// TODO(bcmills): once we have located a plausible module path, we should
	// query version control (if available) to verify that it matches the major
	// version of the most recent tag.
	// See https://golang.org/issue/29433, https://golang.org/issue/27009, and
	// https://golang.org/issue/31549.

	// Cast about for import comments,
	// first in top-level directory, then in subdirectories.
	list, _ := ioutil.ReadDir(dir)
	for _, info := range list {
		if info.Mode().IsRegular() && strings.HasSuffix(info.Name(), ".go") {
			if com := findImportComment(filepath.Join(dir, info.Name())); com != "" {
				return com, nil
			}
		}
	}
	for _, info1 := range list {
		if info1.IsDir() {
			files, _ := ioutil.ReadDir(filepath.Join(dir, info1.Name()))
			for _, info2 := range files {
				if info2.Mode().IsRegular() && strings.HasSuffix(info2.Name(), ".go") {
					if com := findImportComment(filepath.Join(dir, info1.Name(), info2.Name())); com != "" {
						return path.Dir(com), nil
					}
				}
			}
		}
	}

	// Look for Godeps.json declaring import path.
	data, _ := ioutil.ReadFile(filepath.Join(dir, "Godeps/Godeps.json"))
	var cfg1 struct{ ImportPath string }
	json.Unmarshal(data, &cfg1)
	if cfg1.ImportPath != "" {
		return cfg1.ImportPath, nil
	}

	// Look for vendor.json declaring import path.
	data, _ = ioutil.ReadFile(filepath.Join(dir, "vendor/vendor.json"))
	var cfg2 struct{ RootPath string }
	json.Unmarshal(data, &cfg2)
	if cfg2.RootPath != "" {
		return cfg2.RootPath, nil
	}

	// Look for path in GOPATH.
	for _, gpdir := range filepath.SplitList(cfg.BuildContext.GOPATH) {
		if gpdir == "" {
			continue
		}
		if rel := search.InDir(dir, filepath.Join(gpdir, "src")); rel != "" && rel != "." {
			return filepath.ToSlash(rel), nil
		}
	}

	msg := `cannot determine module path for source directory %s (outside GOPATH, module path must be specified)

Example usage:
	'go mod init example.com/m' to initialize a v0 or v1 module
	'go mod init example.com/m/v2' to initialize a v2 module

Run 'go help mod init' for more information.
`
	return "", fmt.Errorf(msg, dir)
}

var (
	importCommentRE = lazyregexp.New(`(?m)^package[ \t]+[^ \t\r\n/]+[ \t]+//[ \t]+import[ \t]+(\"[^"]+\")[ \t]*\r?\n`)
)

func findImportComment(file string) string {
	data, err := ioutil.ReadFile(file)
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

// MinReqs returns a Reqs with minimal additional dependencies of Target,
// as will be written to go.mod.
func MinReqs() mvs.Reqs {
	var retain []string
	for _, m := range buildList[1:] {
		_, explicit := index.require[m]
		if explicit || loaded.direct[m.Path] {
			retain = append(retain, m.Path)
		}
	}
	min, err := mvs.Req(Target, retain, Reqs())
	if err != nil {
		base.Fatalf("go: %v", err)
	}
	return &mvsReqs{buildList: append([]module.Version{Target}, min...)}
}

// WriteGoMod writes the current build list back to go.mod.
func WriteGoMod() {
	// If we're using -mod=vendor we basically ignored
	// go.mod, so definitely don't try to write back our
	// incomplete view of the world.
	if !allowWriteGoMod || cfg.BuildMod == "vendor" {
		return
	}

	// If we aren't in a module, we don't have anywhere to write a go.mod file.
	if modRoot == "" {
		return
	}

	if cfg.BuildMod != "readonly" {
		addGoStmt()
	}

	if loaded != nil {
		reqs := MinReqs()
		min, err := reqs.Required(Target)
		if err != nil {
			base.Fatalf("go: %v", err)
		}
		var list []*modfile.Require
		for _, m := range min {
			list = append(list, &modfile.Require{
				Mod:      m,
				Indirect: !loaded.direct[m.Path],
			})
		}
		modFile.SetRequire(list)
	}
	modFile.Cleanup()

	dirty := index.modFileIsDirty(modFile)
	if dirty && cfg.BuildMod == "readonly" {
		// If we're about to fail due to -mod=readonly,
		// prefer to report a dirty go.mod over a dirty go.sum
		if cfg.BuildModReason != "" {
			base.Fatalf("go: updates to go.mod needed, disabled by -mod=readonly\n\t(%s)", cfg.BuildModReason)
		} else {
			base.Fatalf("go: updates to go.mod needed, disabled by -mod=readonly")
		}
	}
	// Always update go.sum, even if we didn't change go.mod: we may have
	// downloaded modules that we didn't have before.
	modfetch.WriteGoSum()

	if !dirty && cfg.CmdName != "mod tidy" {
		// The go.mod file has the same semantic content that it had before
		// (but not necessarily the same exact bytes).
		// Ignore any intervening edits.
		return
	}

	new, err := modFile.Format()
	if err != nil {
		base.Fatalf("go: %v", err)
	}
	defer func() {
		// At this point we have determined to make the go.mod file on disk equal to new.
		index = indexModFile(new, modFile, false)
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

// indexModFile rebuilds the index of modFile.
// If modFile has been changed since it was first read,
// modFile.Cleanup must be called before indexModFile.
func indexModFile(data []byte, modFile *modfile.File, needsFix bool) *modFileIndex {
	i := new(modFileIndex)
	i.data = data
	i.dataNeedsFix = needsFix

	i.module = module.Version{}
	if modFile.Module != nil {
		i.module = modFile.Module.Mod
	}

	i.goVersion = ""
	if modFile.Go != nil {
		i.goVersion = modFile.Go.Version
	}

	i.require = make(map[module.Version]requireMeta, len(modFile.Require))
	for _, r := range modFile.Require {
		i.require[r.Mod] = requireMeta{indirect: r.Indirect}
	}

	i.replace = make(map[module.Version]module.Version, len(modFile.Replace))
	for _, r := range modFile.Replace {
		if prev, dup := i.replace[r.Old]; dup && prev != r.New {
			base.Fatalf("go: conflicting replacements for %v:\n\t%v\n\t%v", r.Old, prev, r.New)
		}
		i.replace[r.Old] = r.New
	}

	i.exclude = make(map[module.Version]bool, len(modFile.Exclude))
	for _, x := range modFile.Exclude {
		i.exclude[x.Mod] = true
	}

	return i
}

// modFileIsDirty reports whether the go.mod file differs meaningfully
// from what was indexed.
// If modFile has been changed (even cosmetically) since it was first read,
// modFile.Cleanup must be called before modFileIsDirty.
func (i *modFileIndex) modFileIsDirty(modFile *modfile.File) bool {
	if i == nil {
		return modFile != nil
	}

	if i.dataNeedsFix {
		return true
	}

	if modFile.Module == nil {
		if i.module != (module.Version{}) {
			return true
		}
	} else if modFile.Module.Mod != i.module {
		return true
	}

	if modFile.Go == nil {
		if i.goVersion != "" {
			return true
		}
	} else if modFile.Go.Version != i.goVersion {
		if i.goVersion == "" && cfg.BuildMod == "readonly" {
			// go.mod files did not always require a 'go' version, so do not error out
			// if one is missing — we may be inside an older module in the module
			// cache, and should bias toward providing useful behavior.
		} else {
			return true
		}
	}

	if len(modFile.Require) != len(i.require) ||
		len(modFile.Replace) != len(i.replace) ||
		len(modFile.Exclude) != len(i.exclude) {
		return true
	}

	for _, r := range modFile.Require {
		if meta, ok := i.require[r.Mod]; !ok {
			return true
		} else if r.Indirect != meta.indirect {
			if cfg.BuildMod == "readonly" {
				// The module's requirements are consistent; only the "// indirect"
				// comments that are wrong. But those are only guaranteed to be accurate
				// after a "go mod tidy" — it's a good idea to run those before
				// committing a change, but it's certainly not mandatory.
			} else {
				return true
			}
		}
	}

	for _, r := range modFile.Replace {
		if r.New != i.replace[r.Old] {
			return true
		}
	}

	for _, x := range modFile.Exclude {
		if !i.exclude[x.Mod] {
			return true
		}
	}

	return false
}
