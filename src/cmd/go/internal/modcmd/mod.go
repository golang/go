// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package modcmd implements the ``go mod'' command.
package modcmd

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"

	"cmd/go/internal/base"
	"cmd/go/internal/modfile"
	"cmd/go/internal/modload"
	"cmd/go/internal/module"
	"cmd/go/internal/par"
)

var CmdMod = &base.Command{
	UsageLine: "mod [-v] [maintenance flags]",
	Short:     "module maintenance",
	Long: `
Mod performs module maintenance operations as specified by the
following flags, which may be combined.

The -v flag enables additional output about operations performed.

The first group of operations provide low-level editing operations
for manipulating go.mod from the command line or in scripts or
other tools. They read only go.mod itself; they do not look up any
information about the modules involved.

The -init flag initializes and writes a new go.mod to the current directory,
in effect creating a new module rooted at the current directory.
The file go.mod must not already exist.
If possible, mod will guess the module path from import comments
(see 'go help importpath') or from version control configuration.
To override this guess, use the -module flag.
(Without -init, mod applies to the current module.)

The -module flag changes (or, with -init, sets) the module's path
(the go.mod file's module line).

The -require=path@version and -droprequire=path flags
add and drop a requirement on the given module path and version.
Note that -require overrides any existing requirements on path.
These flags are mainly for tools that understand the module graph.
Users should prefer 'go get path@version' or 'go get path@none',
which make other go.mod adjustments as needed to satisfy
constraints imposed by other modules.

The -exclude=path@version and -dropexclude=path@version flags
add and drop an exclusion for the given module path and version.
Note that -exclude=path@version is a no-op if that exclusion already exists.

The -replace=old@v=new@w and -dropreplace=old@v flags
add and drop a replacement of the given module path and version pair.
If the @v in old@v is omitted, the replacement applies to all versions
with the old module path. If the @v in new@v is omitted, the
new path should be a directory on the local system, not a module path.
Note that -replace overrides any existing replacements for old@v.

These editing flags (-require, -droprequire, -exclude, -dropexclude,
-replace, and -dropreplace) may be repeated.

The -fmt flag reformats the go.mod file without making other changes.
This reformatting is also implied by any other modifications that use or
rewrite the go.mod file. The only time this flag is needed is if no other
flags are specified, as in 'go mod -fmt'.

The -graph flag prints the module requirement graph (with replacements applied)
in text form. Each line in the output has two space-separated fields: a module
and one of its requirements. Each module is identified as a string of the form
path@version, except for the main module, which has no @version suffix.

The -json flag prints the go.mod file in JSON format corresponding to these
Go types:

	type Module struct {
		Path string
		Version string
	}

	type GoMod struct {
		Module Module
		Require []Require
		Exclude []Module
		Replace []Replace
	}
	
	type Require struct {
		Path string
		Version string
		Indirect bool
	}
	
	type Replace string {
		Old Module
		New Module
	}

Note that this only describes the go.mod file itself, not other modules
referred to indirectly. For the full set of modules available to a build,
use 'go list -m -json all'.

The next group of operations provide higher-level editing and maintenance
of a module, beyond the go.mod file.

The -packages flag prints a list of packages in the module.
It only identifies directories containing Go source code;
it does not check that those directories contain code that builds.

The -fix flag updates go.mod to use canonical version identifiers and
to be semantically consistent. For example, consider this go.mod file:

	module M

	require (
		A v1
		B v1.0.0
		C v1.0.0
		D v1.2.3
		E dev
	)

	exclude D v1.2.3

First, -fix rewrites non-canonical version identifiers to semver form, so
A's v1 becomes v1.0.0 and E's dev becomes the pseudo-version for the latest
commit on the dev branch, perhaps v0.0.0-20180523231146-b3f5c0f6e5f1.

Next, -fix updates requirements to respect exclusions, so the requirement
on the excluded D v1.2.3 is updated to use the next available version of D,
perhaps D v1.2.4 or D v1.3.0.

Finally, -fix removes redundant or misleading requirements.
For example, if A v1.0.0 itself requires B v1.2.0 and C v1.0.0,
then go.mod's requirement of B v1.0.0 is misleading (superseded
by B's need for v1.2.0), and its requirement of C v1.0.0 is redundant
(implied by B's need for the same version), so both will be removed.

Although -fix runs the fix-up operation in isolation, the fix-up also
runs automatically any time a go command uses the module graph,
to update go.mod to reflect reality. For example, the -sync, -vendor,
and -verify flags all effectively imply -fix. And because the module
graph defines the meaning of import statements, any commands
that load packages—'go build', 'go test', 'go list', and so on—also
effectively imply 'go mod -fix'.

The -sync flag synchronizes go.mod with the source code in the module.
It adds any missing modules necessary to build the current module's
packages and dependencies, and it removes unused modules that
don't provide any relevant packages.

The -vendor flag resets the module's vendor directory to include all
packages needed to build and test all the module's packages.
It does not include any test code for the vendored packages.

The -verify flag checks that the dependencies of the current module,
which are stored in a local downloaded source cache, have not been
modified since being downloaded. If all the modules are unmodified,
-verify prints "all modules verified." Otherwise it reports which
modules have been changed and causes 'go mod' to exit with a
non-zero status.
	`,
}

var (
	modV = CmdMod.Flag.Bool("v", false, "")

	modFmt      = CmdMod.Flag.Bool("fmt", false, "")
	modFix      = CmdMod.Flag.Bool("fix", false, "")
	modGraph    = CmdMod.Flag.Bool("graph", false, "")
	modJSON     = CmdMod.Flag.Bool("json", false, "")
	modPackages = CmdMod.Flag.Bool("packages", false, "")
	modSync     = CmdMod.Flag.Bool("sync", false, "")
	modVendor   = CmdMod.Flag.Bool("vendor", false, "")
	modVerify   = CmdMod.Flag.Bool("verify", false, "")

	modEdits []func(*modfile.File) // edits specified in flags
)

type flagFunc func(string)

func (f flagFunc) String() string     { return "" }
func (f flagFunc) Set(s string) error { f(s); return nil }

func init() {
	CmdMod.Run = runMod // break init cycle

	CmdMod.Flag.BoolVar(&modload.CmdModInit, "init", modload.CmdModInit, "")
	CmdMod.Flag.StringVar(&modload.CmdModModule, "module", modload.CmdModModule, "")

	CmdMod.Flag.Var(flagFunc(flagRequire), "require", "")
	CmdMod.Flag.Var(flagFunc(flagDropRequire), "droprequire", "")
	CmdMod.Flag.Var(flagFunc(flagExclude), "exclude", "")
	CmdMod.Flag.Var(flagFunc(flagDropReplace), "dropreplace", "")
	CmdMod.Flag.Var(flagFunc(flagReplace), "replace", "")
	CmdMod.Flag.Var(flagFunc(flagDropExclude), "dropexclude", "")

	base.AddBuildFlagsNX(&CmdMod.Flag)
}

func runMod(cmd *base.Command, args []string) {
	if modload.Init(); !modload.Enabled() {
		base.Fatalf("go mod: cannot use outside module")
	}
	if len(args) != 0 {
		base.Fatalf("go mod: mod takes no arguments")
	}

	anyFlags :=
		modload.CmdModInit ||
			modload.CmdModModule != "" ||
			*modVendor ||
			*modVerify ||
			*modJSON ||
			*modFmt ||
			*modFix ||
			*modGraph ||
			*modPackages ||
			*modSync ||
			len(modEdits) > 0

	if !anyFlags {
		base.Fatalf("go mod: no flags specified (see 'go help mod').")
	}

	if modload.CmdModModule != "" {
		if err := module.CheckPath(modload.CmdModModule); err != nil {
			base.Fatalf("go mod: invalid -module: %v", err)
		}
	}

	if modload.CmdModInit {
		if _, err := os.Stat("go.mod"); err == nil {
			base.Fatalf("go mod -init: go.mod already exists")
		}
	}
	modload.InitMod()

	// Syntactic edits.

	modFile := modload.ModFile()
	if modload.CmdModModule != "" {
		modFile.AddModuleStmt(modload.CmdModModule)
	}

	if len(modEdits) > 0 {
		for _, edit := range modEdits {
			edit(modFile)
		}
	}
	modFile.SortBlocks()
	modload.WriteGoMod() // write back syntactic changes

	// Semantic edits.

	needBuildList := *modFix || *modGraph

	if *modSync || *modVendor || needBuildList {
		var pkgs []string
		if *modSync || *modVendor {
			pkgs = modload.LoadALL()
		} else {
			modload.LoadBuildList()
		}
		if *modSync {
			// LoadALL already added missing modules.
			// Remove unused modules.
			used := map[module.Version]bool{modload.Target: true}
			for _, pkg := range pkgs {
				used[modload.PackageModule(pkg)] = true
			}

			inGoMod := make(map[string]bool)
			for _, r := range modload.ModFile().Require {
				inGoMod[r.Mod.Path] = true
			}

			var keep []module.Version
			for _, m := range modload.BuildList() {
				if used[m] {
					keep = append(keep, m)
				} else if *modV && inGoMod[m.Path] {
					fmt.Fprintf(os.Stderr, "unused %s\n", m.Path)
				}
			}
			modload.SetBuildList(keep)
		}
		modload.WriteGoMod()
		if *modVendor {
			runVendor()
		}
	}

	// Read-only queries, processed only after updating go.mod.

	if *modJSON {
		modPrintJSON()
	}

	if *modGraph {
		modPrintGraph()
	}

	if *modPackages {
		for _, pkg := range modload.TargetPackages() {
			fmt.Printf("%s\n", pkg)
		}
	}

	if *modVerify {
		runVerify()
	}
}

// parsePathVersion parses -flag=arg expecting arg to be path@version.
func parsePathVersion(flag, arg string) (path, version string) {
	i := strings.Index(arg, "@")
	if i < 0 {
		base.Fatalf("go mod: -%s=%s: need path@version", flag, arg)
	}
	path, version = strings.TrimSpace(arg[:i]), strings.TrimSpace(arg[i+1:])
	if err := module.CheckPath(path); err != nil {
		base.Fatalf("go mod: -%s=%s: invalid path: %v", flag, arg, err)
	}

	// We don't call modfile.CheckPathVersion, because that insists
	// on versions being in semver form, but here we want to allow
	// versions like "master" or "1234abcdef", which the go command will resolve
	// the next time it runs (or during -fix).
	// Even so, we need to make sure the version is a valid token.
	if modfile.MustQuote(version) {
		base.Fatalf("go mod: -%s=%s: invalid version %q", flag, arg, version)
	}

	return path, version
}

// parsePath parses -flag=arg expecting arg to be path (not path@version).
func parsePath(flag, arg string) (path string) {
	if strings.Contains(arg, "@") {
		base.Fatalf("go mod: -%s=%s: need just path, not path@version", flag, arg)
	}
	path = arg
	if err := module.CheckPath(path); err != nil {
		base.Fatalf("go mod: -%s=%s: invalid path: %v", flag, arg, err)
	}
	return path
}

// flagRequire implements the -require flag.
func flagRequire(arg string) {
	path, version := parsePathVersion("require", arg)
	modEdits = append(modEdits, func(f *modfile.File) {
		if err := f.AddRequire(path, version); err != nil {
			base.Fatalf("go mod: -require=%s: %v", arg, err)
		}
	})
}

// flagDropRequire implements the -droprequire flag.
func flagDropRequire(arg string) {
	path := parsePath("droprequire", arg)
	modEdits = append(modEdits, func(f *modfile.File) {
		if err := f.DropRequire(path); err != nil {
			base.Fatalf("go mod: -droprequire=%s: %v", arg, err)
		}
	})
}

// flagExclude implements the -exclude flag.
func flagExclude(arg string) {
	path, version := parsePathVersion("exclude", arg)
	modEdits = append(modEdits, func(f *modfile.File) {
		if err := f.AddExclude(path, version); err != nil {
			base.Fatalf("go mod: -exclude=%s: %v", arg, err)
		}
	})
}

// flagDropExclude implements the -dropexclude flag.
func flagDropExclude(arg string) {
	path, version := parsePathVersion("dropexclude", arg)
	modEdits = append(modEdits, func(f *modfile.File) {
		if err := f.DropExclude(path, version); err != nil {
			base.Fatalf("go mod: -dropexclude=%s: %v", arg, err)
		}
	})
}

// flagReplace implements the -replace flag.
func flagReplace(arg string) {
	var i int
	if i = strings.Index(arg, "="); i < 0 {
		base.Fatalf("go mod: -replace=%s: need old@v=new[@v] (missing =)", arg)
	}
	old, new := strings.TrimSpace(arg[:i]), strings.TrimSpace(arg[i+1:])
	if strings.HasPrefix(new, ">") {
		base.Fatalf("go mod: -replace=%s: separator between old and new is =, not =>", arg)
	}
	var oldPath, oldVersion string
	if i = strings.Index(old, "@"); i < 0 {
		oldPath = old
	} else {
		oldPath, oldVersion = strings.TrimSpace(old[:i]), strings.TrimSpace(old[i+1:])
	}
	if err := module.CheckPath(oldPath); err != nil {
		base.Fatalf("go mod: -replace=%s: invalid old path: %v", arg, err)
	}
	if oldPath != old && modfile.MustQuote(oldVersion) {
		base.Fatalf("go mod: -replace=%s: invalid old version %q", arg, oldVersion)
	}
	var newPath, newVersion string
	if i = strings.Index(new, "@"); i >= 0 {
		newPath, newVersion = strings.TrimSpace(new[:i]), strings.TrimSpace(new[i+1:])
		if err := module.CheckPath(newPath); err != nil {
			base.Fatalf("go mod: -replace=%s: invalid new path: %v", arg, err)
		}
		if modfile.MustQuote(newVersion) {
			base.Fatalf("go mod: -replace=%s: invalid new version %q", arg, newVersion)
		}
	} else {
		if !modfile.IsDirectoryPath(new) {
			base.Fatalf("go mod: -replace=%s: unversioned new path must be local directory", arg)
		}
		newPath = new
	}

	modEdits = append(modEdits, func(f *modfile.File) {
		if err := f.AddReplace(oldPath, oldVersion, newPath, newVersion); err != nil {
			base.Fatalf("go mod: -replace=%s: %v", arg, err)
		}
	})
}

// flagDropReplace implements the -dropreplace flag.
func flagDropReplace(arg string) {
	path, version := parsePathVersion("dropreplace", arg)
	modEdits = append(modEdits, func(f *modfile.File) {
		if err := f.DropReplace(path, version); err != nil {
			base.Fatalf("go mod: -dropreplace=%s: %v", arg, err)
		}
	})
}

// fileJSON is the -json output data structure.
type fileJSON struct {
	Module  module.Version
	Require []requireJSON
	Exclude []module.Version
	Replace []replaceJSON
}

type requireJSON struct {
	Path     string
	Version  string `json:",omitempty"`
	Indirect bool   `json:",omitempty"`
}

type replaceJSON struct {
	Old module.Version
	New module.Version
}

// modPrintJSON prints the -json output.
func modPrintJSON() {
	modFile := modload.ModFile()

	var f fileJSON
	f.Module = modFile.Module.Mod
	for _, r := range modFile.Require {
		f.Require = append(f.Require, requireJSON{Path: r.Mod.Path, Version: r.Mod.Version, Indirect: r.Indirect})
	}
	for _, x := range modFile.Exclude {
		f.Exclude = append(f.Exclude, x.Mod)
	}
	for _, r := range modFile.Replace {
		f.Replace = append(f.Replace, replaceJSON{r.Old, r.New})
	}
	data, err := json.MarshalIndent(&f, "", "\t")
	if err != nil {
		base.Fatalf("go mod -json: internal error: %v", err)
	}
	data = append(data, '\n')
	os.Stdout.Write(data)
}

// modPrintGraph prints the -graph output.
func modPrintGraph() {
	reqs := modload.Reqs()

	format := func(m module.Version) string {
		if m.Version == "" {
			return m.Path
		}
		return m.Path + "@" + m.Version
	}

	// Note: using par.Work only to manage work queue.
	// No parallelism here, so no locking.
	var out []string
	var deps int // index in out where deps start
	var work par.Work
	work.Add(modload.Target)
	work.Do(1, func(item interface{}) {
		m := item.(module.Version)
		list, _ := reqs.Required(m)
		for _, r := range list {
			work.Add(r)
			out = append(out, format(m)+" "+format(r)+"\n")
		}
		if m == modload.Target {
			deps = len(out)
		}
	})

	sort.Slice(out[deps:], func(i, j int) bool {
		return out[deps+i][0] < out[deps+j][0]
	})

	w := bufio.NewWriter(os.Stdout)
	for _, line := range out {
		w.WriteString(line)
	}
	w.Flush()
}
