// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// go mod edit

package modcmd

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"

	"cmd/go/internal/base"
	"cmd/go/internal/modfile"
	"cmd/go/internal/modload"
	"cmd/go/internal/module"
)

var cmdEdit = &base.Command{
	UsageLine: "go mod edit [editing flags] [go.mod]",
	Short:     "edit go.mod from tools or scripts",
	Long: `
Edit provides a command-line interface for editing go.mod,
for use primarily by tools or scripts. It reads only go.mod;
it does not look up information about the modules involved.
By default, edit reads and writes the go.mod file of the main module,
but a different target file can be specified after the editing flags.

The editing flags specify a sequence of editing operations.

The -fmt flag reformats the go.mod file without making other changes.
This reformatting is also implied by any other modifications that use or
rewrite the go.mod file. The only time this flag is needed is if no other
flags are specified, as in 'go mod edit -fmt'.

The -module flag changes the module's path (the go.mod file's module line).

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

The -replace=old[@v]=new[@v] and -dropreplace=old[@v] flags
add and drop a replacement of the given module path and version pair.
If the @v in old@v is omitted, the replacement applies to all versions
with the old module path. If the @v in new@v is omitted, the new path
should be a local module root directory, not a module path.
Note that -replace overrides any existing replacements for old[@v].

The -require, -droprequire, -exclude, -dropexclude, -replace,
and -dropreplace editing flags may be repeated, and the changes
are applied in the order given.

The -go=version flag sets the expected Go language version.

The -print flag prints the final go.mod in its text format instead of
writing it back to go.mod.

The -json flag prints the final go.mod file in JSON format instead of
writing it back to go.mod. The JSON output corresponds to these Go types:

	type Module struct {
		Path string
		Version string
	}

	type GoMod struct {
		Module  Module
		Go      string
		Require []Require
		Exclude []Module
		Replace []Replace
	}

	type Require struct {
		Path string
		Version string
		Indirect bool
	}

	type Replace struct {
		Old Module
		New Module
	}

Note that this only describes the go.mod file itself, not other modules
referred to indirectly. For the full set of modules available to a build,
use 'go list -m -json all'.

For example, a tool can obtain the go.mod as a data structure by
parsing the output of 'go mod edit -json' and can then make changes
by invoking 'go mod edit' with -require, -exclude, and so on.
	`,
}

var (
	editFmt    = cmdEdit.Flag.Bool("fmt", false, "")
	editGo     = cmdEdit.Flag.String("go", "", "")
	editJSON   = cmdEdit.Flag.Bool("json", false, "")
	editPrint  = cmdEdit.Flag.Bool("print", false, "")
	editModule = cmdEdit.Flag.String("module", "", "")
	edits      []func(*modfile.File) // edits specified in flags
)

type flagFunc func(string)

func (f flagFunc) String() string     { return "" }
func (f flagFunc) Set(s string) error { f(s); return nil }

func init() {
	cmdEdit.Run = runEdit // break init cycle

	cmdEdit.Flag.Var(flagFunc(flagRequire), "require", "")
	cmdEdit.Flag.Var(flagFunc(flagDropRequire), "droprequire", "")
	cmdEdit.Flag.Var(flagFunc(flagExclude), "exclude", "")
	cmdEdit.Flag.Var(flagFunc(flagDropReplace), "dropreplace", "")
	cmdEdit.Flag.Var(flagFunc(flagReplace), "replace", "")
	cmdEdit.Flag.Var(flagFunc(flagDropExclude), "dropexclude", "")

	base.AddBuildFlagsNX(&cmdEdit.Flag)
}

func runEdit(cmd *base.Command, args []string) {
	anyFlags :=
		*editModule != "" ||
			*editGo != "" ||
			*editJSON ||
			*editPrint ||
			*editFmt ||
			len(edits) > 0

	if !anyFlags {
		base.Fatalf("go mod edit: no flags specified (see 'go help mod edit').")
	}

	if *editJSON && *editPrint {
		base.Fatalf("go mod edit: cannot use both -json and -print")
	}

	if len(args) > 1 {
		base.Fatalf("go mod edit: too many arguments")
	}
	var gomod string
	if len(args) == 1 {
		gomod = args[0]
	} else {
		modload.MustInit()
		gomod = filepath.Join(modload.ModRoot, "go.mod")
	}

	if *editModule != "" {
		if err := module.CheckPath(*editModule); err != nil {
			base.Fatalf("go mod: invalid -module: %v", err)
		}
	}

	if *editGo != "" {
		if !modfile.GoVersionRE.MatchString(*editGo) {
			base.Fatalf(`go mod: invalid -go option; expecting something like "-go 1.12"`)
		}
	}

	data, err := ioutil.ReadFile(gomod)
	if err != nil {
		base.Fatalf("go: %v", err)
	}

	modFile, err := modfile.Parse(gomod, data, nil)
	if err != nil {
		base.Fatalf("go: errors parsing %s:\n%s", base.ShortPath(gomod), err)
	}

	if *editModule != "" {
		modFile.AddModuleStmt(modload.CmdModModule)
	}

	if *editGo != "" {
		if err := modFile.AddGoStmt(*editGo); err != nil {
			base.Fatalf("go: internal error: %v", err)
		}
	}

	if len(edits) > 0 {
		for _, edit := range edits {
			edit(modFile)
		}
	}
	modFile.SortBlocks()
	modFile.Cleanup() // clean file after edits

	if *editJSON {
		editPrintJSON(modFile)
		return
	}

	data, err = modFile.Format()
	if err != nil {
		base.Fatalf("go: %v", err)
	}

	if *editPrint {
		os.Stdout.Write(data)
		return
	}

	if err := ioutil.WriteFile(gomod, data, 0666); err != nil {
		base.Fatalf("go: %v", err)
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

// parsePathVersionOptional parses path[@version], using adj to
// describe any errors.
func parsePathVersionOptional(adj, arg string, allowDirPath bool) (path, version string, err error) {
	if i := strings.Index(arg, "@"); i < 0 {
		path = arg
	} else {
		path, version = strings.TrimSpace(arg[:i]), strings.TrimSpace(arg[i+1:])
	}
	if err := module.CheckPath(path); err != nil {
		if !allowDirPath || !modfile.IsDirectoryPath(path) {
			return path, version, fmt.Errorf("invalid %s path: %v", adj, err)
		}
	}
	if path != arg && modfile.MustQuote(version) {
		return path, version, fmt.Errorf("invalid %s version: %q", adj, version)
	}
	return path, version, nil
}

// flagRequire implements the -require flag.
func flagRequire(arg string) {
	path, version := parsePathVersion("require", arg)
	edits = append(edits, func(f *modfile.File) {
		if err := f.AddRequire(path, version); err != nil {
			base.Fatalf("go mod: -require=%s: %v", arg, err)
		}
	})
}

// flagDropRequire implements the -droprequire flag.
func flagDropRequire(arg string) {
	path := parsePath("droprequire", arg)
	edits = append(edits, func(f *modfile.File) {
		if err := f.DropRequire(path); err != nil {
			base.Fatalf("go mod: -droprequire=%s: %v", arg, err)
		}
	})
}

// flagExclude implements the -exclude flag.
func flagExclude(arg string) {
	path, version := parsePathVersion("exclude", arg)
	edits = append(edits, func(f *modfile.File) {
		if err := f.AddExclude(path, version); err != nil {
			base.Fatalf("go mod: -exclude=%s: %v", arg, err)
		}
	})
}

// flagDropExclude implements the -dropexclude flag.
func flagDropExclude(arg string) {
	path, version := parsePathVersion("dropexclude", arg)
	edits = append(edits, func(f *modfile.File) {
		if err := f.DropExclude(path, version); err != nil {
			base.Fatalf("go mod: -dropexclude=%s: %v", arg, err)
		}
	})
}

// flagReplace implements the -replace flag.
func flagReplace(arg string) {
	var i int
	if i = strings.Index(arg, "="); i < 0 {
		base.Fatalf("go mod: -replace=%s: need old[@v]=new[@w] (missing =)", arg)
	}
	old, new := strings.TrimSpace(arg[:i]), strings.TrimSpace(arg[i+1:])
	if strings.HasPrefix(new, ">") {
		base.Fatalf("go mod: -replace=%s: separator between old and new is =, not =>", arg)
	}
	oldPath, oldVersion, err := parsePathVersionOptional("old", old, false)
	if err != nil {
		base.Fatalf("go mod: -replace=%s: %v", arg, err)
	}
	newPath, newVersion, err := parsePathVersionOptional("new", new, true)
	if err != nil {
		base.Fatalf("go mod: -replace=%s: %v", arg, err)
	}
	if newPath == new && !modfile.IsDirectoryPath(new) {
		base.Fatalf("go mod: -replace=%s: unversioned new path must be local directory", arg)
	}

	edits = append(edits, func(f *modfile.File) {
		if err := f.AddReplace(oldPath, oldVersion, newPath, newVersion); err != nil {
			base.Fatalf("go mod: -replace=%s: %v", arg, err)
		}
	})
}

// flagDropReplace implements the -dropreplace flag.
func flagDropReplace(arg string) {
	path, version, err := parsePathVersionOptional("old", arg, true)
	if err != nil {
		base.Fatalf("go mod: -dropreplace=%s: %v", arg, err)
	}
	edits = append(edits, func(f *modfile.File) {
		if err := f.DropReplace(path, version); err != nil {
			base.Fatalf("go mod: -dropreplace=%s: %v", arg, err)
		}
	})
}

// fileJSON is the -json output data structure.
type fileJSON struct {
	Module  module.Version
	Go      string `json:",omitempty"`
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

// editPrintJSON prints the -json output.
func editPrintJSON(modFile *modfile.File) {
	var f fileJSON
	f.Module = modFile.Module.Mod
	if modFile.Go != nil {
		f.Go = modFile.Go.Version
	}
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
		base.Fatalf("go: internal error: %v", err)
	}
	data = append(data, '\n')
	os.Stdout.Write(data)
}
