// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Action graph execution methods related to coverage.

package work

import (
	"cmd/go/internal/cfg"
	"cmd/go/internal/str"
	"cmd/internal/cov/covcmd"
	"context"
	"encoding/json"
	"fmt"
	"internal/coverage"
	"io"
	"os"
	"path/filepath"
)

// CovData invokes "go tool covdata" with the specified arguments
// as part of the execution of action 'a'.
func (b *Builder) CovData(a *Action, cmdargs ...any) ([]byte, error) {
	cmdline := str.StringList(cmdargs...)
	args := append([]string{}, cfg.BuildToolexec...)
	args = append(args, "go", "tool", "covdata")
	args = append(args, cmdline...)
	return b.Shell(a).runOut(a.Objdir, nil, args)
}

// BuildActionCoverMetaFile locates and returns the path of the
// meta-data file written by the "go tool cover" step as part of the
// build action for the "go test -cover" run action 'runAct'. Note
// that if the package has no functions the meta-data file will exist
// but will be empty; in this case the return is an empty string.
func BuildActionCoverMetaFile(runAct *Action) (string, error) {
	p := runAct.Package
	barrierAct := runAct.Deps[0]
	for i := range barrierAct.Deps {
		pred := barrierAct.Deps[i]
		if pred.Mode != "build" || pred.Package == nil {
			continue
		}
		if pred.Package.ImportPath == p.ImportPath {
			metaFile := pred.Objdir + covcmd.MetaFileForPackage(p.ImportPath)
			if cfg.BuildN {
				return metaFile, nil
			}
			f, err := os.Open(metaFile)
			if err != nil {
				return "", err
			}
			defer f.Close()
			fi, err2 := f.Stat()
			if err2 != nil {
				return "", err2
			}
			if fi.Size() == 0 {
				return "", nil
			}
			return metaFile, nil
		}
	}
	return "", fmt.Errorf("internal error: unable to locate build action for package %q run action", p.ImportPath)
}

// WriteCoveragePercent writes out to the writer 'w' a "percent
// statements covered" for the package whose test-run action is
// 'runAct', based on the meta-data file 'mf'. This helper is used in
// cases where a user runs "go test -cover" on a package that has
// functions but no tests; in the normal case (package has tests)
// the percentage is written by the test binary when it runs.
func WriteCoveragePercent(b *Builder, runAct *Action, mf string, w io.Writer) error {
	dir := filepath.Dir(mf)
	output, cerr := b.CovData(runAct, "percent", "-i", dir)
	if cerr != nil {
		return b.Shell(runAct).reportCmd("", "", output, cerr)
	}
	_, werr := w.Write(output)
	return werr
}

// WriteCoverageProfile writes out a coverage profile fragment for the
// package whose test-run action is 'runAct'; content is written to
// the file 'outf' based on the coverage meta-data info found in
// 'mf'. This helper is used in cases where a user runs "go test
// -cover" on a package that has functions but no tests.
func WriteCoverageProfile(b *Builder, runAct *Action, mf, outf string, w io.Writer) error {
	dir := filepath.Dir(mf)
	output, err := b.CovData(runAct, "textfmt", "-i", dir, "-o", outf)
	if err != nil {
		return b.Shell(runAct).reportCmd("", "", output, err)
	}
	_, werr := w.Write(output)
	return werr
}

// WriteCoverMetaFilesFile writes out a summary file ("meta-files
// file") as part of the action function for the "writeCoverMeta"
// pseudo action employed during "go test -coverpkg" runs where there
// are multiple tests and multiple packages covered. It builds up a
// table mapping package import path to meta-data file fragment and
// writes it out to a file where it can be read by the various test
// run actions. Note that this function has to be called A) after the
// build actions are complete for all packages being tested, and B)
// before any of the "run test" actions for those packages happen.
// This requirement is enforced by adding making this action ("a")
// dependent on all test package build actions, and making all test
// run actions dependent on this action.
func WriteCoverMetaFilesFile(b *Builder, ctx context.Context, a *Action) error {
	sh := b.Shell(a)

	// Build the metafilecollection object.
	var collection coverage.MetaFileCollection
	for i := range a.Deps {
		dep := a.Deps[i]
		if dep.Mode != "build" {
			panic("unexpected mode " + dep.Mode)
		}
		metaFilesFile := dep.Objdir + covcmd.MetaFileForPackage(dep.Package.ImportPath)
		// Check to make sure the meta-data file fragment exists
		//  and has content (may be empty if package has no functions).
		if fi, err := os.Stat(metaFilesFile); err != nil {
			continue
		} else if fi.Size() == 0 {
			continue
		}
		collection.ImportPaths = append(collection.ImportPaths, dep.Package.ImportPath)
		collection.MetaFileFragments = append(collection.MetaFileFragments, metaFilesFile)
	}

	// Serialize it.
	data, err := json.Marshal(collection)
	if err != nil {
		return fmt.Errorf("marshal MetaFileCollection: %v", err)
	}
	data = append(data, '\n') // makes -x output more readable

	// Create the directory for this action's objdir and
	// then write out the serialized collection
	// to a file in the directory.
	if err := sh.Mkdir(a.Objdir); err != nil {
		return err
	}
	mfpath := a.Objdir + coverage.MetaFilesFileName
	if err := sh.writeFile(mfpath, data); err != nil {
		return fmt.Errorf("writing metafiles file: %v", err)
	}

	// We're done.
	return nil
}
