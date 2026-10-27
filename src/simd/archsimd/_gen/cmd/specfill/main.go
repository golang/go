// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// specfill checks the public SIMD API against internal/spec and reports
// signature mismatches, comment structure violations, and documentation diffs.
package main

import (
	"bytes"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"simd/archsimd/_gen/gentools"
	"simd/archsimd/_gen/specdoc"
	"simd/archsimd/_gen/specgen"
	"strings"
)

func main() {
	genOpts := gentools.RegisterFlags(nil)
	noFillDoc := flag.Bool("no-fill-doc", false, "disable filling doc comments from spec")
	noFillNames := flag.Bool("no-fill-names", false, "disable filling parameter and result names from spec")

	dirs := []string{
		filepath.Join(genOpts.GOROOT, "src/simd"),
		filepath.Join(genOpts.GOROOT, "src/simd/archsimd"),
	}

	flag.Usage = func() {
		w := flag.CommandLine.Output()
		fmt.Fprintf(w, "usage: specfill [flags] [files...]\n\n")
		fmt.Fprintf(w, "If files are not given, specfill applies to all non-test, non-generated files in:\n")
		for _, d := range dirs {
			fmt.Fprintf(w, "  %s\n", d)
		}
		fmt.Fprintln(w)
		flag.CommandLine.PrintDefaults()
	}

	flag.Parse()

	specDir := specgen.MustFindSpecDir(genOpts.GOROOT)
	specFuncs, err := specgen.Load(specDir, nil)
	if err != nil {
		fmt.Fprintf(os.Stderr, "loading spec: %v\n", err)
		os.Exit(1)
	}
	specIdx := specgen.NewIndex(specFuncs)

	var targetFiles []string
	filesArgs := false
	if flag.NArg() > 0 {
		filesArgs = true
		for _, arg := range flag.Args() {
			p, err := filepath.Abs(arg)
			if err != nil {
				fmt.Fprintf(os.Stderr, "%s: %v\n", arg, err)
				os.Exit(1)
			}
			targetFiles = append(targetFiles, p)
		}
	} else {
		for _, dir := range dirs {
			entries, err := os.ReadDir(dir)
			if err != nil {
				fmt.Fprintf(os.Stderr, "reading %s: %v\n", dir, err)
				os.Exit(1)
			}
			for _, e := range entries {
				if e.IsDir() || !strings.HasSuffix(e.Name(), ".go") || strings.HasSuffix(e.Name(), "_test.go") {
					continue
				}
				targetFiles = append(targetFiles, filepath.Join(dir, e.Name()))
			}
		}
	}

	var files gentools.Files
	defer files.FlushOrExit()

	var combinedReport specdoc.Report

	for _, filePath := range targetFiles {
		relPath, err := filepath.Rel(genOpts.GOROOT, filePath)
		if err != nil {
			relPath = filePath
		}
		srcRelPath, err := filepath.Rel(filepath.Join(genOpts.GOROOT, "src"), filePath)
		if err != nil {
			srcRelPath = strings.TrimPrefix(filepath.ToSlash(relPath), "src/")
		}
		src, err := os.ReadFile(filePath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "reading %s: %v\n", filePath, err)
			os.Exit(1)
		}
		gen, err := isGenerated(src)
		if err != nil {
			fmt.Fprintf(os.Stderr, "%s: %v\n", filePath, err)
			os.Exit(1)
		}
		if gen {
			if filesArgs {
				fmt.Fprintf(os.Stderr, "%s: file is generated\n", filePath)
				os.Exit(1)
			}
			continue
		}
		rewritten, err := specdoc.Fill(src, specIdx, specdoc.Options{
			Filename:        filepath.ToSlash(relPath),
			AllowDocRewrite: true,
			NoFillDoc:       *noFillDoc,
			NoFillNames:     *noFillNames,
		})
		var rep *specdoc.Report
		if errors.As(err, &rep) {
			combinedReport.Merge(rep)
		} else if err != nil {
			fmt.Fprintf(os.Stderr, "checking %s: %v\n", filePath, err)
			os.Exit(1)
		}
		if rewritten != nil && !bytes.Equal(src, rewritten) {
			buf := files.NewGoFile(filepath.ToSlash(srcRelPath))
			buf.Write(rewritten)
		}
	}

	combinedReport.Print(os.Stderr)

	if genOpts.Write {
		if err := checkCleanWorkingCopy(genOpts.GOROOT); err != nil {
			fmt.Fprintf(os.Stderr, "refusing to write: %v\n", err)
			os.Exit(1)
		}
		if !combinedReport.Empty() {
			fmt.Fprintln(os.Stderr, "errors reported; not writing files")
			os.Exit(1)
		}
	}
}

// checkCleanWorkingCopy verifies that the working directory has no uncommitted
// changes before specfill rewrites files in place (-w).
//
// We rely on "git status --porcelain" rather than invoking "jj" commands directly.
// In a standard Git repository or a colocated jj/git repository, "git status"
// checks disk cleanliness without side-effects.
//
// # Interoperation with colocated jj repos
//
// Jujutsu keeps Git's HEAD and index in sync with the parent of the working-copy
// commit (@-). If the user starts from a fresh, empty commit (e.g. via "jj new"),
// the working copy matches Git's HEAD and git status reports clean. If there are
// modified or untracked files on disk (or if @ contains changes), git status
// reports them as uncommitted changes. By querying git rather than running "jj"
// commands (like "jj diff" or "jj status"), we avoid jj's snapshotting side-effects,
// such as automatically snapshotting dirty files into @, writing to the operation
// log, and auto-tracking newly created files.
func checkCleanWorkingCopy(dir string) error {
	cmd := exec.Command("git", "status", "--porcelain")
	cmd.Dir = dir
	out, err := cmd.Output()
	if err != nil {
		// Not a git repository or git binary not available; skip the check.
		return nil
	}
	if len(bytes.TrimSpace(out)) > 0 {
		return fmt.Errorf("working copy has uncommitted changes:\n%s", out)
	}
	return nil
}

// isGenerated reports whether src is a generated Go file.
//
// It checks for the standard "// Code generated by" comment marker at the beginning
// of the file or at the beginning of any line. If missing, it checks for
// "DO NOT EDIT" as a backstop; if found, it returns an error to catch non-standard
// or malformed generator headers rather than erroneously treating the file as hand-written.
func isGenerated(src []byte) (bool, error) {
	const marker = "// Code generated by"
	if bytes.HasPrefix(src, []byte(marker)) || bytes.Contains(src, []byte("\n"+marker)) {
		return true, nil
	}
	if bytes.Contains(src, []byte("DO NOT EDIT")) {
		return false, errors.New("file contains 'DO NOT EDIT' but lacks '// Code generated by' header")
	}
	return false, nil
}
