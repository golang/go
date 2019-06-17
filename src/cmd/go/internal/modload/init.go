// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modload

import (
	"bytes"
	"encoding/json"
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
	"cmd/go/internal/modconv"
	"cmd/go/internal/modfetch"
	"cmd/go/internal/modfetch/codehost"
	"cmd/go/internal/modfile"
	"cmd/go/internal/module"
	"cmd/go/internal/mvs"
	"cmd/go/internal/renameio"
	"cmd/go/internal/search"
)

var (
	cwd            string // TODO(bcmills): Is this redundant with base.Cwd?
	mustUseModules = false
	initialized    bool

	modRoot     string
	modFile     *modfile.File
	modFileData []byte
	excluded    map[module.Version]bool
	Target      module.Version

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

	var err error
	cwd, err = os.Getwd()
	if err != nil {
		base.Fatalf("go: %v", err)
	}

	if CmdModInit {
		// Running 'go mod init': go.mod will be created in current directory.
		modRoot = cwd
	} else {
		modRoot = findModuleRoot(cwd)
		if modRoot == "" {
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
		// If the command is 'go get' or 'go list' and all of the args are in the
		// same existing module, we could use that module's download directory in
		// the module cache as the module root, applying any replacements and/or
		// exclusions specified by that module. However, that would leave us in a
		// strange state: we want 'go get' to be consistent with 'go list', and 'go
		// list' should be able to operate on multiple modules. Moreover, the 'get'
		// target might specify relative file paths (e.g. in the same repository) as
		// replacements, and we would not be able to apply those anyway: we would
		// need to either error out or ignore just those replacements, when a build
		// from an empty module could proceed without error.
		//
		// Instead, we'll operate as though we're in some ephemeral external module,
		// ignoring all replacements and exclusions uniformly.

		// Normally we check sums using the go.sum file from the main module, but
		// without a main module we do not have an authoritative go.sum file.
		//
		// TODO(bcmills): In Go 1.13, check sums when outside the main module.
		//
		// One possible approach is to merge the go.sum files from all of the
		// modules we download: that doesn't protect us against bad top-level
		// modules, but it at least ensures consistency for transitive dependencies.
	} else {
		modfetch.GoSumFile = filepath.Join(modRoot, "go.sum")
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
	if cwd != "" {
		if dir, name := findAltConfig(cwd); dir != "" {
			rel, err := filepath.Rel(cwd, dir)
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
	base.Fatalf("go: cannot find main module; see 'go help modules'")
}

// InitMod sets Target and, if there is a main module, parses the initial build
// list from its go.mod file, creating and populating that file if needed.
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

	gomod := filepath.Join(modRoot, "go.mod")
	data, err := renameio.ReadFile(gomod)
	if err != nil {
		base.Fatalf("go: %v", err)
	}

	f, err := modfile.Parse(gomod, data, fixVersion)
	if err != nil {
		// Errors returned by modfile.Parse begin with file:line.
		base.Fatalf("go: errors parsing go.mod:\n%s\n", err)
	}
	modFile = f
	modFileData = data

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

	excluded = make(map[module.Version]bool)
	for _, x := range f.Exclude {
		excluded[x.Mod] = true
	}
	modFileToBuildList()
	stdVendorMode()
	WriteGoMod()
}

// modFileToBuildList initializes buildList from the modFile.
func modFileToBuildList() {
	Target = modFile.Module.Mod
	targetPrefix = Target.Path
	if rel := search.InDir(cwd, cfg.GOROOTsrc); rel != "" {
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

// stdVendorMode applies inside $GOROOT/src.
// It checks that the go.mod matches vendor/modules.txt
// and then sets -mod=vendor unless this is a command
// that has to do explicitly with modules.
func stdVendorMode() {
	if !targetInGorootSrc {
		return
	}
	if cfg.CmdName == "get" || strings.HasPrefix(cfg.CmdName, "mod ") {
		return
	}

	readVendorList()
BuildList:
	for _, m := range buildList {
		if m.Path == "cmd" || m.Path == "std" {
			continue
		}
		for _, v := range vendorList {
			if m.Path == v.Path {
				if m.Version != v.Version {
					base.Fatalf("go: inconsistent vendoring in %s:\n"+
						"\tgo.mod requires %s %s but vendor/modules.txt has %s.\n"+
						"\trun 'go mod tidy; go mod vendor' to sync",
						modRoot, m.Path, m.Version, v.Version)
				}
				continue BuildList
			}
		}
		base.Fatalf("go: inconsistent vendoring in %s:\n"+
			"\tgo.mod requires %s %s but vendor/modules.txt does not include it.\n"+
			"\trun 'go mod tidy; go mod vendor' to sync", modRoot, m.Path, m.Version)
	}
	cfg.BuildMod = "vendor"
}

// Allowed reports whether module m is allowed (not excluded) by the main module's go.mod.
func Allowed(m module.Version) bool {
	return !excluded[m]
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

// MinReqs returns a Reqs with minimal dependencies of Target,
// as will be written to go.mod.
func MinReqs() mvs.Reqs {
	var direct []string
	for _, m := range buildList[1:] {
		if loaded.direct[m.Path] {
			direct = append(direct, m.Path)
		}
	}
	min, err := mvs.Req(Target, buildList, direct, Reqs())
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

	addGoStmt()

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

	modFile.Cleanup() // clean file after edits
	new, err := modFile.Format()
	if err != nil {
		base.Fatalf("go: %v", err)
	}

	dirty := !bytes.Equal(new, modFileData)
	if dirty && cfg.BuildMod == "readonly" {
		// If we're about to fail due to -mod=readonly,
		// prefer to report a dirty go.mod over a dirty go.sum
		base.Fatalf("go: updates to go.mod needed, disabled by -mod=readonly")
	}
	// Always update go.sum, even if we didn't change go.mod: we may have
	// downloaded modules that we didn't have before.
	modfetch.WriteGoSum()

	if !dirty {
		// We don't need to modify go.mod from what we read previously.
		// Ignore any intervening edits.
		return
	}

	unlock := modfetch.SideLock()
	defer unlock()

	file := filepath.Join(modRoot, "go.mod")
	old, err := renameio.ReadFile(file)
	if !bytes.Equal(old, modFileData) {
		if bytes.Equal(old, new) {
			// Some other process wrote the same go.mod file that we were about to write.
			modFileData = new
			return
		}
		if err != nil {
			base.Fatalf("go: can't determine whether go.mod has changed: %v", err)
		}
		// The contents of the go.mod file have changed. In theory we could add all
		// of the new modules to the build list, recompute, and check whether any
		// module in *our* build list got bumped to a different version, but that's
		// a lot of work for marginal benefit. Instead, fail the command: if users
		// want to run concurrent commands, they need to start with a complete,
		// consistent module definition.
		base.Fatalf("go: updates to go.mod needed, but contents have changed")

	}

	if err := renameio.WriteFile(file, new, 0666); err != nil {
		base.Fatalf("error writing go.mod: %v", err)
	}
	modFileData = new
}

func fixVersion(path, vers string) (string, error) {
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
		if err := module.MatchPathMajor(vers, pathMajor); err == nil {
			return vers, nil
		}
	}

	info, err := Query(path, vers, "", nil)
	if err != nil {
		return "", err
	}
	return info.Version, nil
}
