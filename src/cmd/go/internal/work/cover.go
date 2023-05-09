// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Action graph execution methods related to coverage.

package work

import (
	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/str"
	"fmt"
	"internal/coverage/covcmd"
	"io"
	"os"
	"path/filepath"
)

// CovData invokes "go tool covdata" with the specified arguments
// as part of the execution of action 'a'.
func (b *Builder) CovData(a *Action, cmdargs ...any) ([]byte, error) {
	cmdline := str.StringList(cmdargs...)
	args := append([]string{}, cfg.BuildToolexec...)
	args = append(args, base.Tool("covdata"))
	args = append(args, cmdline...)
	return b.runOut(a, a.Objdir, nil, args)
}

// BuildActionCoverMetaFile locates and returns the path of the
// meta-data file written by the "go tool cover" step as part of the
// build action for the "go test -cover" run action 'runAct'. Note
// that if the package has no functions the meta-data file will exist
// but will be empty; in this case the return is an empty string.
func BuildActionCoverMetaFile(runAct *Action) (string, error) {
	p := runAct.Package
	for i := range runAct.Deps {
		pred := runAct.Deps[i]
		if pred.Mode != "build" || pred.Package == nil {
			continue
		}
		if pred.Package.ImportPath == p.ImportPath {
			metaFile := pred.Objdir + covcmd.MetaFileForPackage(p.ImportPath)
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
		p := runAct.Package
		return formatOutput(b.WorkDir, p.Dir, p.ImportPath,
			p.Desc(), string(output))
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
		p := runAct.Package
		return formatOutput(b.WorkDir, p.Dir, p.ImportPath,
			p.Desc(), string(output))
	}
	_, werr := w.Write(output)
	return werr
}
