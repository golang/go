// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// go work use

package workcmd

import (
	"context"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"

	"cmd/go/internal/base"
	"cmd/go/internal/fsys"
	"cmd/go/internal/gover"
	"cmd/go/internal/modload"
	"cmd/go/internal/str"
	"cmd/go/internal/toolchain"

	"golang.org/x/mod/modfile"
)

var cmdUse = &base.Command{
	UsageLine: "go work use [-r] [moddirs]",
	Short:     "add modules to workspace file",
	Long: `Use provides a command-line interface for adding
directories, optionally recursively, to a go.work file.

A use directive will be added to the go.work file for each argument
directory listed on the command line go.work file, if it exists,
or removed from the go.work file if it does not exist.
Use fails if any remaining use directives refer to modules that
do not exist.

Use updates the go line in go.work to specify a version at least as
new as all the go lines in the used modules, both preexisting ones
and newly added ones. With no arguments, this update is the only
thing that go work use does.

The -r flag searches recursively for modules in the argument
directories, and the use command operates as if each of the directories
were specified as arguments.



See the workspaces reference at https://go.dev/ref/mod#workspaces
for more information.
`,
}

var useR = cmdUse.Flag.Bool("r", false, "")

func init() {
	cmdUse.Run = runUse // break init cycle

	base.AddChdirFlag(&cmdUse.Flag)
	base.AddModCommonFlags(&cmdUse.Flag)
}

func runUse(ctx context.Context, cmd *base.Command, args []string) {
	moduleLoaderState := modload.NewState()
	moduleLoaderState.ForceUseModules = true
	modload.InitWorkfile(moduleLoaderState)
	gowork := modload.WorkFilePath(moduleLoaderState)
	if gowork == "" {
		base.Fatalf("go: no go.work file found\n\t(run 'go work init' first or specify path using GOWORK environment variable)")
	}
	wf, err := modload.ReadWorkFile(gowork)
	if err != nil {
		base.Fatal(err)
	}
	workUse(ctx, moduleLoaderState, gowork, wf, args)
	modload.WriteWorkFile(gowork, wf)
}

func workUse(ctx context.Context, s *modload.State, gowork string, wf *modfile.WorkFile, args []string) {
	workDir := filepath.Dir(gowork) // absolute, since gowork itself is absolute

	haveDirs := make(map[string][]string) // absolute → original(s)
	for _, use := range wf.Use {
		var abs string
		if filepath.IsAbs(use.Path) {
			abs = filepath.Clean(use.Path)
		} else {
			abs = filepath.Join(workDir, use.Path)
		}
		haveDirs[abs] = append(haveDirs[abs], use.Path)
	}

	// keepDirs maps each absolute path to keep to the literal string to use for
	// that path (either an absolute or a relative path), or the empty string if
	// all entries for the absolute path should be removed.
	keepDirs := make(map[string]string)

	sw := toolchain.NewSwitcher(s)

	// lookDir updates the entry in keepDirs for the directory dir,
	// which is either absolute or relative to the current working directory
	// (not necessarily the directory containing the workfile).
	lookDir := func(dir string) {
		absDir, dir := pathRel(workDir, dir)

		file := filepath.Join(absDir, "go.mod")
		fi, err := fsys.Stat(file)
		if err != nil {
			if os.IsNotExist(err) {
				keepDirs[absDir] = ""
			} else {
				sw.Error(err)
			}
			return
		}

		if !fi.Mode().IsRegular() {
			sw.Error(fmt.Errorf("%v is not a regular file", base.ShortPath(file)))
			return
		}

		if dup := keepDirs[absDir]; dup != "" && dup != dir {
			base.Errorf(`go: already added "%s" as "%s"`, dir, dup)
		}
		keepDirs[absDir] = dir
	}

	for _, useDir := range args {
		absArg, _ := pathRel(workDir, useDir)

		info, err := fsys.Stat(absArg)
		if err != nil {
			// Errors raised from os.Stat are formatted to be more user-friendly.
			if os.IsNotExist(err) {
				err = fmt.Errorf("directory %v does not exist", base.ShortPath(absArg))
			}
			sw.Error(err)
			continue
		} else if !info.IsDir() {
			sw.Error(fmt.Errorf("%s is not a directory", base.ShortPath(absArg)))
			continue
		}

		if !*useR {
			lookDir(useDir)
			continue
		}

		// Add or remove entries for any subdirectories that still exist.
		// If the root itself is a symlink to a directory,
		// we want to follow it (see https://go.dev/issue/50807).
		// Add a trailing separator to force that to happen.
		fsys.WalkDir(str.WithFilePathSeparator(useDir), func(path string, d fs.DirEntry, err error) error {
			if err != nil {
				return err
			}

			if !d.IsDir() {
				if d.Type()&fs.ModeSymlink != 0 {
					if target, err := fsys.Stat(path); err == nil && target.IsDir() {
						fmt.Fprintf(os.Stderr, "warning: ignoring symlink %s\n", base.ShortPath(path))
					}
				}
				return nil
			}
			lookDir(path)
			return nil
		})

		// Remove entries for subdirectories that no longer exist.
		// Because they don't exist, they will be skipped by Walk.
		for absDir := range haveDirs {
			if str.HasFilePathPrefix(absDir, absArg) {
				if _, ok := keepDirs[absDir]; !ok {
					keepDirs[absDir] = "" // Mark for deletion.
				}
			}
		}
	}

	// Update the work file.
	for absDir, keepDir := range keepDirs {
		nKept := 0
		for _, dir := range haveDirs[absDir] {
			if dir == keepDir { // (note that dir is always non-empty)
				nKept++
			} else {
				wf.DropUse(dir)
			}
		}
		if keepDir != "" && nKept != 1 {
			// If we kept more than one copy, delete them all.
			// We'll recreate a unique copy with AddUse.
			if nKept > 1 {
				wf.DropUse(keepDir)
			}
			wf.AddUse(keepDir, "")
		}
	}

	// Read the Go versions from all the use entries, old and new (but not dropped).
	goV := gover.FromGoWork(wf)
	for _, use := range wf.Use {
		if use.Path == "" { // deleted
			continue
		}
		var abs string
		if filepath.IsAbs(use.Path) {
			abs = filepath.Clean(use.Path)
		} else {
			abs = filepath.Join(workDir, use.Path)
		}
		_, mf, err := modload.ReadModFile(filepath.Join(abs, "go.mod"), nil)
		if err != nil {
			sw.Error(err)
			continue
		}
		goV = gover.Max(goV, gover.FromGoMod(mf))
	}
	sw.Switch(ctx)
	base.ExitIfErrors()

	modload.UpdateWorkGoVersion(wf, goV)
	modload.UpdateWorkFile(wf)
}

// pathRel returns the absolute and canonical forms of dir for use in a
// go.work file located in directory workDir.
//
// If dir is relative, it is interpreted relative to base.Cwd()
// and its canonical form is relative to workDir if possible.
// If dir is absolute or cannot be made relative to workDir,
// its canonical form is absolute.
//
// Canonical absolute paths are clean.
// Canonical relative paths are clean and slash-separated.
func pathRel(workDir, dir string) (abs, canonical string) {
	if filepath.IsAbs(dir) {
		abs = filepath.Clean(dir)
		return abs, abs
	}

	abs = filepath.Join(base.Cwd(), dir)
	rel, err := filepath.Rel(workDir, abs)
	if err != nil {
		// The path can't be made relative to the go.work file,
		// so it must be kept absolute instead.
		return abs, abs
	}

	// Normalize relative paths to use slashes, so that checked-in go.work
	// files with relative paths within the repo are platform-independent.
	return abs, modload.ToDirectoryPath(rel)
}
