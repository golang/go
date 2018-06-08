// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vgo

import (
	"bytes"
	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/modconv"
	"cmd/go/internal/modfetch"
	"cmd/go/internal/modfetch/codehost"
	"cmd/go/internal/modfile"
	"cmd/go/internal/module"
	"cmd/go/internal/mvs"
	"cmd/go/internal/search"
	"cmd/go/internal/semver"
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

var (
	cwd         string
	enabled     = MustBeVgo
	MustBeVgo   = mustBeVgo()
	initialized bool

	ModRoot  string
	modFile  *modfile.File
	excluded map[module.Version]bool
	Target   module.Version

	gopath string
	srcV   string
)

func BinDir() string {
	if !Enabled() {
		panic("vgo.Bin")
	}
	return filepath.Join(gopath, "bin")
}

func init() {
	flag.BoolVar(&MustBeVgo, "vgo", MustBeVgo, "require use of modules")
}

// mustBeVgo reports whether we are invoked as vgo
// (as opposed to go).
// If so, we only support builds with go.mod files.
func mustBeVgo() bool {
	name := os.Args[0]
	name = name[strings.LastIndex(name, "/")+1:]
	name = name[strings.LastIndex(name, `\`)+1:]
	return strings.HasPrefix(name, "vgo")
}

func Init() {
	if initialized {
		return
	}
	initialized = true

	// If this is testgo - the test binary during cmd/go tests - then
	// do not let it look for a go.mod. Only use vgo support if the
	// global -vgo flag has been passed on the command line.
	if base := filepath.Base(os.Args[0]); (base == "testgo" || base == "testgo.exe") && !MustBeVgo {
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

	root, _ := FindModuleRoot(cwd, "", MustBeVgo)
	if root == "" {
		// If invoked as vgo, insist on a mod file.
		if MustBeVgo {
			base.Fatalf("cannot determine module root; please create a go.mod file there")
		}
		return
	}
	enabled = true
	ModRoot = root
	search.SetModRoot(root)
}

func Enabled() bool {
	return false // COMPLETELY OFF FOR NOW
	/*
		if !initialized {
			panic("vgo: Enabled called before Init")
		}
		return enabled
	*/
}

func InitMod() {
	if Init(); !Enabled() || modFile != nil {
		return
	}

	list := filepath.SplitList(cfg.BuildContext.GOPATH)
	if len(list) == 0 || list[0] == "" {
		base.Fatalf("missing $GOPATH")
	}
	gopath = list[0]
	if _, err := os.Stat(filepath.Join(gopath, "go.mod")); err == nil {
		base.Fatalf("$GOPATH/go.mod exists but should not")
	}
	srcV = filepath.Join(list[0], "src/v")
	codehost.WorkRoot = filepath.Join(srcV, "cache/vcswork")

	gomod := filepath.Join(ModRoot, "go.mod")
	data, err := ioutil.ReadFile(gomod)
	if err != nil {
		legacyModInit()
		return
	}

	f, err := modfile.Parse(gomod, data, fixVersion)
	if err != nil {
		// Errors returned by modfile.Parse begin with file:line.
		base.Fatalf("vgo: errors parsing go.mod:\n%s\n", err)
	}
	modFile = f

	if len(f.Syntax.Stmt) == 0 || f.Module == nil {
		// Empty mod file. Must add module path.
		path, err := FindModulePath(ModRoot)
		if err != nil {
			base.Fatalf("vgo: %v", err)
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
	Target = f.Module.Mod
	writeGoMod()
}

func allowed(m module.Version) bool {
	return !excluded[m]
}

func legacyModInit() {
	if modFile == nil {
		path, err := FindModulePath(ModRoot)
		if err != nil {
			base.Fatalf("vgo: %v", err)
		}
		fmt.Fprintf(os.Stderr, "vgo: creating new go.mod: module %s\n", path)
		modFile = new(modfile.File)
		modFile.AddModuleStmt(path)
	}

	Target = modFile.Module.Mod
	for _, name := range altConfigs {
		cfg := filepath.Join(ModRoot, name)
		data, err := ioutil.ReadFile(cfg)
		if err == nil {
			convert := modconv.Converters[name]
			if convert == nil {
				return
			}
			fmt.Fprintf(os.Stderr, "vgo: copying requirements from %s\n", cfg)
			cfg = filepath.ToSlash(cfg)
			if err := modfetch.ConvertLegacyConfig(modFile, cfg, data); err != nil {
				base.Fatalf("vgo: %v", err)
			}
			if len(modFile.Syntax.Stmt) == 1 {
				// Add comment to prevent vgo from re-converting every time it runs.
				modFile.AddComment("// vgo: no requirements found in " + name)
			}
			return
		}
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

// Exported only for testing.
func FindModuleRoot(dir, limit string, legacyConfigOK bool) (root, file string) {
	dir = filepath.Clean(dir)
	dir1 := dir
	limit = filepath.Clean(limit)

	// Look for enclosing go.mod.
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir, "go.mod"
		}
		if dir == limit {
			break
		}
		d := filepath.Dir(dir)
		if d == dir {
			break
		}
		dir = d
	}

	// Failing that, look for enclosing alternate version config.
	if legacyConfigOK {
		dir = dir1
		for {
			for _, name := range altConfigs {
				if _, err := os.Stat(filepath.Join(dir, name)); err == nil {
					return dir, name
				}
			}
			if dir == limit {
				break
			}
			d := filepath.Dir(dir)
			if d == dir {
				break
			}
			dir = d
		}
	}

	return "", ""
}

// Exported only for testing.
func FindModulePath(dir string) (string, error) {
	for _, gpdir := range filepath.SplitList(cfg.BuildContext.GOPATH) {
		src := filepath.Join(gpdir, "src") + string(filepath.Separator)
		if strings.HasPrefix(dir, src) {
			return filepath.ToSlash(dir[len(src):]), nil
		}
	}

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
	var cfg struct{ ImportPath string }
	json.Unmarshal(data, &cfg)
	if cfg.ImportPath != "" {
		return cfg.ImportPath, nil
	}

	// Look for vendor.json declaring import path.
	data, _ = ioutil.ReadFile(filepath.Join(dir, "vendor/vendor.json"))
	var cfg2 struct{ RootPath string }
	json.Unmarshal(data, &cfg2)
	if cfg2.RootPath != "" {
		return cfg2.RootPath, nil
	}

	// Look for .git/config with github origin as last resort.
	data, _ = ioutil.ReadFile(filepath.Join(dir, ".git/config"))
	if m := gitOriginRE.FindSubmatch(data); m != nil {
		return "github.com/" + string(m[1]), nil
	}

	return "", fmt.Errorf("cannot determine module path for source directory %s (outside GOPATH, no import comments)", dir)
}

var (
	gitOriginRE     = regexp.MustCompile(`(?m)^\[remote "origin"\]\r?\n\turl = (?:https://github.com/|git@github.com:|gh:)([^/]+/[^/]+?)(\.git)?\r?\n`)
	importCommentRE = regexp.MustCompile(`(?m)^package[ \t]+[^ \t\r\n/]+[ \t]+//[ \t]+import[ \t]+(\"[^"]+\")[ \t]*\r?\n`)
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

func writeGoMod() {
	writeModHash()

	if buildList != nil {
		min, err := mvs.Req(Target, buildList, newReqs())
		if err != nil {
			base.Fatalf("vgo: %v", err)
		}
		modFile.SetRequire(min)
	}

	file := filepath.Join(ModRoot, "go.mod")
	old, _ := ioutil.ReadFile(file)
	new, err := modFile.Format()
	if err != nil {
		base.Fatalf("vgo: %v", err)
	}
	if bytes.Equal(old, new) {
		return
	}
	if err := ioutil.WriteFile(file, new, 0666); err != nil {
		base.Fatalf("vgo: %v", err)
	}
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
		return "", fmt.Errorf("malformed module path: %s", path)
	}
	if semver.IsValid(vers) && vers == semver.Canonical(vers) && module.MatchPathMajor(vers, pathMajor) {
		return vers, nil
	}

	info, err := modfetch.Query(path, vers, nil)
	if err != nil {
		return "", err
	}
	return info.Version, nil
}
