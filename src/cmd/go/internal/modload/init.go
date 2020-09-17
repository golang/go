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
	"io/ioutil"
	"os"
	"path"
	"path/filepath"
	"runtime/debug"
	"strconv"
	"strings"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/lockedfile"
	"cmd/go/internal/modconv"
	"cmd/go/internal/modfetch"
	"cmd/go/internal/mvs"
	"cmd/go/internal/search"

	"golang.org/x/mod/modfile"
	"golang.org/x/mod/module"
	"golang.org/x/mod/semver"
)

var (
	initialized bool

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

	// RootMode determines whether a module root is needed.
	RootMode Root

	// ForceUseModules may be set to force modules to be enabled when
	// GO111MODULE=auto or to report an error when GO111MODULE=off.
	ForceUseModules bool

	allowMissingModuleImports bool
)

type Root int

const (
	// AutoRoot is the default for most commands. modload.Init will look for
	// a go.mod file in the current directory or any parent. If none is found,
	// modules may be disabled (GO111MODULE=on) or commands may run in a
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
	var mustUseModules bool
	env := cfg.Getenv("GO111MODULE")
	switch env {
	default:
		base.Fatalf("go: unknown environment setting GO111MODULE=%s", env)
	case "auto", "":
		mustUseModules = ForceUseModules
	case "on":
		mustUseModules = true
	case "off":
		if ForceUseModules {
			base.Fatalf("go: modules disabled by GO111MODULE=off; see 'go help modules'")
		}
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
	} else if RootMode == NoRoot {
		// TODO(jayconrod): report an error if -mod -modfile is explicitly set on
		// the command line. Ignore those flags if they come from GOFLAGS.
		modRoot = ""
	} else {
		modRoot = findModuleRoot(base.Cwd)
		if modRoot == "" {
			if cfg.ModFile != "" {
				base.Fatalf("go: cannot find main module, but -modfile was set.\n\t-modfile cannot be used to set the module root directory.")
			}
			if RootMode == NeedRoot {
				base.Fatalf("go: cannot find main module; see 'go help modules'")
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

	// We're in module mode. Install the hooks to make it work.

	list := filepath.SplitList(cfg.BuildContext.GOPATH)
	if len(list) == 0 || list[0] == "" {
		base.Fatalf("missing $GOPATH")
	}
	gopath = list[0]
	if _, err := os.Stat(filepath.Join(gopath, "go.mod")); err == nil {
		base.Fatalf("$GOPATH/go.mod exists but should not")
	}

	cfg.ModulesEnabled = true
	// load.ModDirImportPath = DirImportPath

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
// list from its go.mod file. If InitMod is called by 'go mod init', InitMod
// will populate go.mod in memory, possibly importing dependencies from a
// legacy configuration file. For other commands, InitMod may make other
// adjustments in memory, like adding a go directive. WriteGoMod should be
// called later to write changes out to disk.
//
// As a side-effect, InitMod sets a default for cfg.BuildMod if it does not
// already have an explicit value.
func InitMod(ctx context.Context) {
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
		return
	}

	gomod := ModFilePath()
	data, err := lockedfile.Read(gomod)
	if err != nil {
		base.Fatalf("go: %v", err)
	}

	var fixed bool
	f, err := modfile.Parse(gomod, data, fixVersion(ctx, &fixed))
	if err != nil {
		// Errors returned by modfile.Parse begin with file:line.
		base.Fatalf("go: errors parsing go.mod:\n%s\n", err)
	}
	modFile = f
	index = indexModFile(data, f, fixed)

	if f.Module == nil {
		// No module declaration. Must add module path.
		base.Fatalf("go: no module declaration in go.mod.\n\tRun 'go mod edit -module=example.com/mod' to specify the module path.")
	}

	if len(f.Syntax.Stmt) == 1 && f.Module != nil {
		// Entire file is just a module statement.
		// Populate require if possible.
		legacyModInit()
	}

	if err := checkModulePathLax(f.Module.Mod.Path); err != nil {
		base.Fatalf("go: %v", err)
	}

	setDefaultBuildMod()
	modFileToBuildList()
	if cfg.BuildMod == "vendor" {
		readVendorList()
		checkVendorConsistency()
	}
}

// checkModulePathLax checks that the path meets some minimum requirements
// to avoid confusing users or the module cache. The requirements are weaker
// than those of module.CheckPath to allow room for weakening module path
// requirements in the future, but strong enough to help users avoid significant
// problems.
func checkModulePathLax(p string) error {
	// TODO(matloob): Replace calls of this function in this CL with calls
	// to module.CheckImportPath once it's been laxened, if it becomes laxened.
	// See golang.org/issue/29101 for a discussion about whether to make CheckImportPath
	// more lax or more strict.

	errorf := func(format string, args ...interface{}) error {
		return fmt.Errorf("invalid module path %q: %s", p, fmt.Sprintf(format, args...))
	}

	// Disallow shell characters " ' * < > ? ` | to avoid triggering bugs
	// with file systems and subcommands. Disallow file path separators : and \
	// because path separators other than / will confuse the module cache.
	// See fileNameOK in golang.org/x/mod/module/module.go.
	shellChars := "`" + `\"'*<>?|`
	fsChars := `\:`
	if i := strings.IndexAny(p, shellChars); i >= 0 {
		return errorf("contains disallowed shell character %q", p[i])
	}
	if i := strings.IndexAny(p, fsChars); i >= 0 {
		return errorf("contains disallowed path separator character %q", p[i])
	}

	// Ensure path.IsAbs and build.IsLocalImport are false, and that the path is
	// invariant under path.Clean, also to avoid confusing the module cache.
	if path.IsAbs(p) {
		return errorf("is an absolute path")
	}
	if build.IsLocalImport(p) {
		return errorf("is a local import path")
	}
	if path.Clean(p) != p {
		return errorf("is not clean")
	}

	return nil
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
			if err := module.CheckPathMajor(vers, pathMajor); err == nil {
				return vers, nil
			}
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
		if index != nil && index.exclude[r.Mod] {
			if cfg.BuildMod == "mod" {
				fmt.Fprintf(os.Stderr, "go: dropping requirement on excluded version %s %s\n", r.Mod.Path, r.Mod.Version)
			} else {
				fmt.Fprintf(os.Stderr, "go: ignoring requirement on excluded version %s %s\n", r.Mod.Path, r.Mod.Version)
			}
		} else {
			list = append(list, r.Mod)
		}
	}
	buildList = list
}

// setDefaultBuildMod sets a default value for cfg.BuildMod
// if it is currently empty.
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
		cfg.BuildMod = "readonly"
		return
	}

	if fi, err := os.Stat(filepath.Join(modRoot, "vendor")); err == nil && fi.IsDir() {
		modGo := "unspecified"
		if index.goVersionV != "" {
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
	if rel := search.InDir(dir, cfg.BuildContext.GOROOT); rel != "" {
		// Don't suggest creating a module from $GOROOT/.git/config
		// or a config file found in any parent of $GOROOT (see #34191).
		return "", ""
	}
	for {
		for _, name := range altConfigs {
			if fi, err := os.Stat(filepath.Join(dir, name)); err == nil && !fi.IsDir() {
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
	var badPathErr error
	for _, gpdir := range filepath.SplitList(cfg.BuildContext.GOPATH) {
		if gpdir == "" {
			continue
		}
		if rel := search.InDir(dir, filepath.Join(gpdir, "src")); rel != "" && rel != "." {
			path := filepath.ToSlash(rel)
			// TODO(matloob): replace this with module.CheckImportPath
			// once it's been laxened.
			// Only checkModulePathLax here. There are some unpublishable
			// module names that are compatible with checkModulePathLax
			// but they already work in GOPATH so don't break users
			// trying to do a build with modules. gorelease will alert users
			// publishing their modules to fix their paths.
			if err := checkModulePathLax(path); err != nil {
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
		if cfg.BuildModExplicit {
			base.Fatalf("go: updates to go.mod needed, disabled by -mod=readonly")
		} else if cfg.BuildModReason != "" {
			base.Fatalf("go: updates to go.mod needed, disabled by -mod=readonly\n\t(%s)", cfg.BuildModReason)
		} else {
			base.Fatalf("go: updates to go.mod needed; try 'go mod tidy' first")
		}
	}

	if !dirty && cfg.CmdName != "mod tidy" {
		// The go.mod file has the same semantic content that it had before
		// (but not necessarily the same exact bytes).
		// Don't write go.mod, but write go.sum in case we added or trimmed sums.
		modfetch.WriteGoSum(keepSums(true))
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
		modfetch.WriteGoSum(keepSums(true))
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

// keepSums returns a set of module sums to preserve in go.sum. The set
// includes entries for all modules used to load packages (according to
// the last load function like ImportPaths, LoadALL, etc.). It also contains
// entries for go.mod files needed for MVS (the version of these entries
// ends with "/go.mod").
//
// If addDirect is true, the set also includes sums for modules directly
// required by go.mod, as represented by the index, with replacements applied.
func keepSums(addDirect bool) map[module.Version]bool {
	// Walk the module graph and keep sums needed by MVS.
	modkey := func(m module.Version) module.Version {
		return module.Version{Path: m.Path, Version: m.Version + "/go.mod"}
	}
	keep := make(map[module.Version]bool)
	replaced := make(map[module.Version]bool)
	reqs := Reqs()
	var walk func(module.Version)
	walk = func(m module.Version) {
		// If we build using a replacement module, keep the sum for the replacement,
		// since that's the code we'll actually use during a build.
		r := Replacement(m)
		if r.Path == "" {
			keep[modkey(m)] = true
		} else {
			replaced[m] = true
			keep[modkey(r)] = true
		}
		list, _ := reqs.Required(m)
		for _, r := range list {
			if !keep[modkey(r)] && !replaced[r] {
				walk(r)
			}
		}
	}
	walk(Target)

	// Add entries for modules that provided packages loaded with ImportPaths,
	// LoadALL, or similar functions.
	if loaded != nil {
		for _, pkg := range loaded.pkgs {
			m := pkg.mod
			if r := Replacement(m); r.Path != "" {
				keep[r] = true
			} else {
				keep[m] = true
			}
		}
	}

	// Add entries for modules directly required by go.mod.
	if addDirect {
		for m := range index.require {
			var kept module.Version
			if r := Replacement(m); r.Path != "" {
				kept = r
			} else {
				kept = m
			}
			keep[kept] = true
			keep[module.Version{Path: kept.Path, Version: kept.Version + "/go.mod"}] = true
		}
	}

	return keep
}

func TrimGoSum() {
	// Don't retain sums for direct requirements in go.mod. When TrimGoSum is
	// called, go.mod has not been updated, and it may contain requirements on
	// modules deleted from the build list.
	addDirect := false
	modfetch.TrimGoSum(keepSums(addDirect))
}
