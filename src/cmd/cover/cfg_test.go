// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main_test

import (
	"encoding/json"
	"fmt"
	"internal/coverage"
	"internal/testenv"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func writeFile(t *testing.T, path string, contents []byte) {
	if err := os.WriteFile(path, contents, 0666); err != nil {
		t.Fatalf("os.WriteFile(%s) failed: %v", path, err)
	}
}

func writePkgConfig(t *testing.T, outdir, tag, ppath, pname string, gran string) string {
	incfg := filepath.Join(outdir, tag+"incfg.txt")
	outcfg := filepath.Join(outdir, "outcfg.txt")
	p := coverage.CoverPkgConfig{
		PkgPath:     ppath,
		PkgName:     pname,
		Granularity: gran,
		OutConfig:   outcfg,
	}
	data, err := json.Marshal(p)
	if err != nil {
		t.Fatalf("json.Marshal failed: %v", err)
	}
	writeFile(t, incfg, data)
	return incfg
}

func writeOutFileList(t *testing.T, infiles []string, outdir, tag string) ([]string, string) {
	outfilelist := filepath.Join(outdir, tag+"outfilelist.txt")
	var sb strings.Builder
	outfs := []string{}
	for _, inf := range infiles {
		base := filepath.Base(inf)
		of := filepath.Join(outdir, tag+".cov."+base)
		outfs = append(outfs, of)
		fmt.Fprintf(&sb, "%s\n", of)
	}
	if err := os.WriteFile(outfilelist, []byte(sb.String()), 0666); err != nil {
		t.Fatalf("writing %s: %v", outfilelist, err)
	}
	return outfs, outfilelist
}

func runPkgCover(t *testing.T, outdir string, tag string, incfg string, mode string, infiles []string, errExpected bool) ([]string, string, string) {
	// Write the pkgcfg file.
	outcfg := filepath.Join(outdir, "outcfg.txt")

	// Form up the arguments and run the tool.
	outfiles, outfilelist := writeOutFileList(t, infiles, outdir, tag)
	args := []string{"-pkgcfg", incfg, "-mode=" + mode, "-var=var" + tag, "-outfilelist", outfilelist}
	args = append(args, infiles...)
	cmd := testenv.Command(t, testcover(t), args...)
	if errExpected {
		errmsg := runExpectingError(cmd, t)
		return nil, "", errmsg
	} else {
		run(cmd, t)
		return outfiles, outcfg, ""
	}
}

// Set to true when debugging unit test (to inspect debris, etc).
// Note that this functionality does not work on windows.
const debugWorkDir = false

func TestCoverWithCfg(t *testing.T) {
	testenv.MustHaveGoRun(t)

	t.Parallel()

	// Subdir in testdata that has our input files of interest.
	tpath := filepath.Join("testdata", "pkgcfg")

	// Helper to collect input paths (go files) for a subdir in 'pkgcfg'
	pfiles := func(subdir string) []string {
		de, err := os.ReadDir(filepath.Join(tpath, subdir))
		if err != nil {
			t.Fatalf("reading subdir %s: %v", subdir, err)
		}
		paths := []string{}
		for _, e := range de {
			if !strings.HasSuffix(e.Name(), ".go") || strings.HasSuffix(e.Name(), "_test.go") {
				continue
			}
			paths = append(paths, filepath.Join(tpath, subdir, e.Name()))
		}
		return paths
	}

	dir := tempDir(t)
	if debugWorkDir {
		dir = "/tmp/qqq"
		os.RemoveAll(dir)
		os.Mkdir(dir, 0777)
	}
	instdira := filepath.Join(dir, "insta")
	if err := os.Mkdir(instdira, 0777); err != nil {
		t.Fatal(err)
	}

	scenarios := []struct {
		mode, gran string
	}{
		{
			mode: "count",
			gran: "perblock",
		},
		{
			mode: "set",
			gran: "perfunc",
		},
		{
			mode: "regonly",
			gran: "perblock",
		},
	}

	var incfg string
	for _, scenario := range scenarios {
		// Instrument package "a", producing a set of instrumented output
		// files and an 'output config' file to pass on to the compiler.
		ppath := "cfg/a"
		pname := "a"
		mode := scenario.mode
		gran := scenario.gran
		tag := mode + "_" + gran
		incfg = writePkgConfig(t, instdira, tag, ppath, pname, gran)
		ofs, outcfg, _ := runPkgCover(t, instdira, tag, incfg, mode,
			pfiles("a"), false)
		t.Logf("outfiles: %+v\n", ofs)

		// Run the compiler on the files to make sure the result is
		// buildable.
		bargs := []string{"tool", "compile", "-p", "a", "-coveragecfg", outcfg}
		bargs = append(bargs, ofs...)
		cmd := testenv.Command(t, testenv.GoToolPath(t), bargs...)
		cmd.Dir = instdira
		run(cmd, t)
	}

	// Do some error testing to ensure that various bad options and
	// combinations are properly rejected.

	// Expect error if config file inaccessible/unreadable.
	mode := "atomic"
	errExpected := true
	tag := "errors"
	_, _, errmsg := runPkgCover(t, instdira, tag, "/not/a/file", mode,
		pfiles("a"), errExpected)
	want := "error reading pkgconfig file"
	if !strings.Contains(errmsg, want) {
		t.Errorf("'bad config file' test: wanted %s got %s", want, errmsg)
	}

	// Expect err if config file contains unknown stuff.
	t.Logf("mangling in config")
	writeFile(t, incfg, []byte("blah=foo\n"))
	_, _, errmsg = runPkgCover(t, instdira, tag, incfg, mode,
		pfiles("a"), errExpected)
	want = "error reading pkgconfig file"
	if !strings.Contains(errmsg, want) {
		t.Errorf("'bad config file' test: wanted %s got %s", want, errmsg)
	}

	// Expect error on empty config file.
	t.Logf("writing empty config")
	writeFile(t, incfg, []byte("\n"))
	_, _, errmsg = runPkgCover(t, instdira, tag, incfg, mode,
		pfiles("a"), errExpected)
	if !strings.Contains(errmsg, want) {
		t.Errorf("'bad config file' test: wanted %s got %s", want, errmsg)
	}
}
