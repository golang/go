// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// go work edit

package workcmd

import (
	"bytes"
	"cmd/go/internal/base"
	"cmd/go/internal/lockedfile"
	"cmd/go/internal/modload"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"golang.org/x/mod/module"

	"golang.org/x/mod/modfile"
)

var cmdEdit = &base.Command{
	UsageLine: "go work edit [editing flags] [go.work]",
	Short:     "edit go.work from tools or scripts",
	Long: `Editwork provides a command-line interface for editing go.work,
for use primarily by tools or scripts. It only reads go.work;
it does not look up information about the modules involved.
If no file is specified, editwork looks for a go.work file in the current
directory and its parent directories

The editing flags specify a sequence of editing operations.

The -fmt flag reformats the go.work file without making other changes.
This reformatting is also implied by any other modifications that use or
rewrite the go.mod file. The only time this flag is needed is if no other
flags are specified, as in 'go mod editwork -fmt'.

The -directory=path and -dropdirectory=path flags
add and drop a directory from the go.work files set of module directories.

The -replace=old[@v]=new[@v] flag adds a replacement of the given
module path and version pair. If the @v in old@v is omitted, a
replacement without a version on the left side is added, which applies
to all versions of the old module path. If the @v in new@v is omitted,
the new path should be a local module root directory, not a module
path. Note that -replace overrides any redundant replacements for old[@v],
so omitting @v will drop existing replacements for specific versions.

The -dropreplace=old[@v] flag drops a replacement of the given
module path and version pair. If the @v is omitted, a replacement without
a version on the left side is dropped.

The -directory, -dropdirectory, -replace, and -dropreplace,
editing flags may be repeated, and the changes are applied in the order given.

The -go=version flag sets the expected Go language version.

The -print flag prints the final go.work in its text format instead of
writing it back to go.mod.

The -json flag prints the final go.work file in JSON format instead of
writing it back to go.mod. The JSON output corresponds to these Go types:

	type Module struct {
		Path    string
		Version string
	}

	type GoWork struct {
		Go        string
		Directory []Directory
		Replace   []Replace
	}

	type Directory struct {
		Path       string
		ModulePath string
	}

	type Replace struct {
		Old Module
		New Module
	}

See the workspaces design proposal at
https://go.googlesource.com/proposal/+/master/design/45713-workspace.md for
more information.
`,
}

var (
	editFmt   = cmdEdit.Flag.Bool("fmt", false, "")
	editGo    = cmdEdit.Flag.String("go", "", "")
	editJSON  = cmdEdit.Flag.Bool("json", false, "")
	editPrint = cmdEdit.Flag.Bool("print", false, "")
	workedits []func(file *modfile.WorkFile) // edits specified in flags
)

type flagFunc func(string)

func (f flagFunc) String() string     { return "" }
func (f flagFunc) Set(s string) error { f(s); return nil }

func init() {
	cmdEdit.Run = runEditwork // break init cycle

	cmdEdit.Flag.Var(flagFunc(flagEditworkDirectory), "directory", "")
	cmdEdit.Flag.Var(flagFunc(flagEditworkDropDirectory), "dropdirectory", "")
	cmdEdit.Flag.Var(flagFunc(flagEditworkReplace), "replace", "")
	cmdEdit.Flag.Var(flagFunc(flagEditworkDropReplace), "dropreplace", "")

	base.AddWorkfileFlag(&cmdEdit.Flag)
}

func runEditwork(ctx context.Context, cmd *base.Command, args []string) {
	anyFlags :=
		*editGo != "" ||
			*editJSON ||
			*editPrint ||
			*editFmt ||
			len(workedits) > 0

	if !anyFlags {
		base.Fatalf("go: no flags specified (see 'go help mod editwork').")
	}

	if *editJSON && *editPrint {
		base.Fatalf("go: cannot use both -json and -print")
	}

	if len(args) > 1 {
		base.Fatalf("go: 'go mod editwork' accepts at most one argument")
	}
	var gowork string
	if len(args) == 1 {
		gowork = args[0]
	} else {
		modload.InitWorkfile()
		gowork = modload.WorkFilePath()
	}

	if *editGo != "" {
		if !modfile.GoVersionRE.MatchString(*editGo) {
			base.Fatalf(`go mod: invalid -go option; expecting something like "-go %s"`, modload.LatestGoVersion())
		}
	}

	data, err := lockedfile.Read(gowork)
	if err != nil {
		base.Fatalf("go: %v", err)
	}

	workFile, err := modfile.ParseWork(gowork, data, nil)
	if err != nil {
		base.Fatalf("go: errors parsing %s:\n%s", base.ShortPath(gowork), err)
	}

	if *editGo != "" {
		if err := workFile.AddGoStmt(*editGo); err != nil {
			base.Fatalf("go: internal error: %v", err)
		}
	}

	if len(workedits) > 0 {
		for _, edit := range workedits {
			edit(workFile)
		}
	}
	workFile.SortBlocks()
	workFile.Cleanup() // clean file after edits

	if *editJSON {
		editPrintJSON(workFile)
		return
	}

	out := modfile.Format(workFile.Syntax)

	if *editPrint {
		os.Stdout.Write(out)
		return
	}

	err = lockedfile.Transform(gowork, func(lockedData []byte) ([]byte, error) {
		if !bytes.Equal(lockedData, data) {
			return nil, errors.New("go.work changed during editing; not overwriting")
		}
		return out, nil
	})
	if err != nil {
		base.Fatalf("go: %v", err)
	}
}

// flagEditworkDirectory implements the -directory flag.
func flagEditworkDirectory(arg string) {
	workedits = append(workedits, func(f *modfile.WorkFile) {
		_, mf, err := modload.ReadModFile(filepath.Join(arg, "go.mod"), nil)
		modulePath := ""
		if err == nil {
			modulePath = mf.Module.Mod.Path
		}
		f.AddDirectory(modload.ToDirectoryPath(arg), modulePath)
		if err := f.AddDirectory(modload.ToDirectoryPath(arg), ""); err != nil {
			base.Fatalf("go: -directory=%s: %v", arg, err)
		}
	})
}

// flagEditworkDropDirectory implements the -dropdirectory flag.
func flagEditworkDropDirectory(arg string) {
	workedits = append(workedits, func(f *modfile.WorkFile) {
		if err := f.DropDirectory(modload.ToDirectoryPath(arg)); err != nil {
			base.Fatalf("go: -dropdirectory=%s: %v", arg, err)
		}
	})
}

// allowedVersionArg returns whether a token may be used as a version in go.mod.
// We don't call modfile.CheckPathVersion, because that insists on versions
// being in semver form, but here we want to allow versions like "master" or
// "1234abcdef", which the go command will resolve the next time it runs (or
// during -fix).  Even so, we need to make sure the version is a valid token.
func allowedVersionArg(arg string) bool {
	return !modfile.MustQuote(arg)
}

// parsePathVersionOptional parses path[@version], using adj to
// describe any errors.
func parsePathVersionOptional(adj, arg string, allowDirPath bool) (path, version string, err error) {
	if i := strings.Index(arg, "@"); i < 0 {
		path = arg
	} else {
		path, version = strings.TrimSpace(arg[:i]), strings.TrimSpace(arg[i+1:])
	}
	if err := module.CheckImportPath(path); err != nil {
		if !allowDirPath || !modfile.IsDirectoryPath(path) {
			return path, version, fmt.Errorf("invalid %s path: %v", adj, err)
		}
	}
	if path != arg && !allowedVersionArg(version) {
		return path, version, fmt.Errorf("invalid %s version: %q", adj, version)
	}
	return path, version, nil
}

// flagReplace implements the -replace flag.
func flagEditworkReplace(arg string) {
	var i int
	if i = strings.Index(arg, "="); i < 0 {
		base.Fatalf("go: -replace=%s: need old[@v]=new[@w] (missing =)", arg)
	}
	old, new := strings.TrimSpace(arg[:i]), strings.TrimSpace(arg[i+1:])
	if strings.HasPrefix(new, ">") {
		base.Fatalf("go: -replace=%s: separator between old and new is =, not =>", arg)
	}
	oldPath, oldVersion, err := parsePathVersionOptional("old", old, false)
	if err != nil {
		base.Fatalf("go: -replace=%s: %v", arg, err)
	}
	newPath, newVersion, err := parsePathVersionOptional("new", new, true)
	if err != nil {
		base.Fatalf("go: -replace=%s: %v", arg, err)
	}
	if newPath == new && !modfile.IsDirectoryPath(new) {
		base.Fatalf("go: -replace=%s: unversioned new path must be local directory", arg)
	}

	workedits = append(workedits, func(f *modfile.WorkFile) {
		if err := f.AddReplace(oldPath, oldVersion, newPath, newVersion); err != nil {
			base.Fatalf("go: -replace=%s: %v", arg, err)
		}
	})
}

// flagDropReplace implements the -dropreplace flag.
func flagEditworkDropReplace(arg string) {
	path, version, err := parsePathVersionOptional("old", arg, true)
	if err != nil {
		base.Fatalf("go: -dropreplace=%s: %v", arg, err)
	}
	workedits = append(workedits, func(f *modfile.WorkFile) {
		if err := f.DropReplace(path, version); err != nil {
			base.Fatalf("go: -dropreplace=%s: %v", arg, err)
		}
	})
}

type replaceJSON struct {
	Old module.Version
	New module.Version
}

// editPrintJSON prints the -json output.
func editPrintJSON(workFile *modfile.WorkFile) {
	var f workfileJSON
	if workFile.Go != nil {
		f.Go = workFile.Go.Version
	}
	for _, d := range workFile.Directory {
		f.Directory = append(f.Directory, directoryJSON{DiskPath: d.Path, ModPath: d.ModulePath})
	}

	for _, r := range workFile.Replace {
		f.Replace = append(f.Replace, replaceJSON{r.Old, r.New})
	}
	data, err := json.MarshalIndent(&f, "", "\t")
	if err != nil {
		base.Fatalf("go: internal error: %v", err)
	}
	data = append(data, '\n')
	os.Stdout.Write(data)
}

// workfileJSON is the -json output data structure.
type workfileJSON struct {
	Go        string `json:",omitempty"`
	Directory []directoryJSON
	Replace   []replaceJSON
}

type directoryJSON struct {
	DiskPath string
	ModPath  string `json:",omitempty"`
}
