// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// go work use

package workcmd

import (
	"cmd/go/internal/base"
	"cmd/go/internal/fsys"
	"cmd/go/internal/modload"
	"cmd/go/internal/str"
	"context"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
)

var cmdUse = &base.Command{
	UsageLine: "go work use [-r] moddirs",
	Short:     "add modules to workspace file",
	Long: `Use provides a command-line interface for adding
directories, optionally recursively, to a go.work file.

A use directive will be added to the go.work file for each argument
directory listed on the command line go.work file, if it exists on disk,
or removed from the go.work file if it does not exist on disk.

The -r flag searches recursively for modules in the argument
directories, and the use command operates as if each of the directories
were specified as arguments: namely, use directives will be added for
directories that exist, and removed for directories that do not exist.

See the workspaces reference at https://go.dev/ref/mod#workspaces
for more information.
`,
}

var useR = cmdUse.Flag.Bool("r", false, "")

func init() {
	cmdUse.Run = runUse // break init cycle

	base.AddModCommonFlags(&cmdUse.Flag)
}

func runUse(ctx context.Context, cmd *base.Command, args []string) {
	modload.ForceUseModules = true

	var gowork string
	modload.InitWorkfile()
	gowork = modload.WorkFilePath()

	if gowork == "" {
		base.Fatalf("go: no go.work file found\n\t(run 'go work init' first or specify path using GOWORK environment variable)")
	}
	workFile, err := modload.ReadWorkFile(gowork)
	if err != nil {
		base.Fatalf("go: %v", err)
	}
	workDir := filepath.Dir(gowork) // Absolute, since gowork itself is absolute.

	haveDirs := make(map[string][]string) // absolute → original(s)
	for _, use := range workFile.Use {
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

	// lookDir updates the entry in keepDirs for the directory dir,
	// which is either absolute or relative to the current working directory
	// (not necessarily the directory containing the workfile).
	lookDir := func(dir string) {
		absDir, dir := pathRel(workDir, dir)

		fi, err := fsys.Stat(filepath.Join(absDir, "go.mod"))
		if err != nil {
			if os.IsNotExist(err) {
				keepDirs[absDir] = ""
			} else {
				base.Errorf("go: %v", err)
			}
			return
		}

		if !fi.Mode().IsRegular() {
			base.Errorf("go: %v is not regular", filepath.Join(dir, "go.mod"))
		}

		if dup := keepDirs[absDir]; dup != "" && dup != dir {
			base.Errorf(`go: already added "%s" as "%s"`, dir, dup)
		}
		keepDirs[absDir] = dir
	}

	if len(args) == 0 {
		base.Fatalf("go: 'go work use' requires one or more directory arguments")
	}
	for _, useDir := range args {
		absArg, _ := pathRel(workDir, useDir)

		info, err := fsys.Stat(absArg)
		if err != nil {
			// Errors raised from os.Stat are formatted to be more user-friendly.
			if os.IsNotExist(err) {
				base.Errorf("go: directory %v does not exist", absArg)
			} else {
				base.Errorf("go: %v", err)
			}
			continue
		} else if !info.IsDir() {
			base.Errorf("go: %s is not a directory", absArg)
			continue
		}

		if !*useR {
			lookDir(useDir)
			continue
		}

		// Add or remove entries for any subdirectories that still exist.
		fsys.Walk(useDir, func(path string, info fs.FileInfo, err error) error {
			if err != nil {
				return err
			}

			if !info.IsDir() {
				if info.Mode()&fs.ModeSymlink != 0 {
					if target, err := fsys.Stat(path); err == nil && target.IsDir() {
						fmt.Fprintf(os.Stderr, "warning: ignoring symlink %s\n", path)
					}
				}
				return nil
			}
			lookDir(path)
			return nil
		})

		// Remove entries for subdirectories that no longer exist.
		// Because they don't exist, they will be skipped by Walk.
		for absDir, _ := range haveDirs {
			if str.HasFilePathPrefix(absDir, absArg) {
				if _, ok := keepDirs[absDir]; !ok {
					keepDirs[absDir] = "" // Mark for deletion.
				}
			}
		}
	}

	base.ExitIfErrors()

	for absDir, keepDir := range keepDirs {
		nKept := 0
		for _, dir := range haveDirs[absDir] {
			if dir == keepDir { // (note that dir is always non-empty)
				nKept++
			} else {
				workFile.DropUse(dir)
			}
		}
		if keepDir != "" && nKept != 1 {
			// If we kept more than one copy, delete them all.
			// We'll recreate a unique copy with AddUse.
			if nKept > 1 {
				workFile.DropUse(keepDir)
			}
			workFile.AddUse(keepDir, "")
		}
	}
	modload.UpdateWorkFile(workFile)
	modload.WriteWorkFile(gowork, workFile)
}

// pathRel returns the absolute and canonical forms of dir for use in a
// go.work file located in directory workDir.
//
// If dir is relative, it is intepreted relative to base.Cwd()
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
