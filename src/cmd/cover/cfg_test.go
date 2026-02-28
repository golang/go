// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main_test

import (
	"cmd/internal/cov/covcmd"
	"encoding/json"
	"fmt"
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

func writePkgConfig(t *testing.T, outdir, tag, ppath, pname string, gran string, mpath string) string {
	incfg := filepath.Join(outdir, tag+"incfg.txt")
	outcfg := filepath.Join(outdir, "outcfg.txt")
	p := covcmd.CoverPkgConfig{
		PkgPath:      ppath,
		PkgName:      pname,
		Granularity:  gran,
		OutConfig:    outcfg,
		EmitMetaFile: mpath,
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
	cv := filepath.Join(outdir, "covervars.go")
	outfs := []string{cv}
	fmt.Fprintf(&sb, "%s\n", cv)
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

func TestCoverWithCfg(t *testing.T) {
	testenv.MustHaveGoRun(t)

	t.Parallel()

	// Subdir in testdata that has our input files of interest.
	tpath := filepath.Join("testdata", "pkgcfg")
	dir := tempDir(t)
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
	apkgfiles := []string{filepath.Join(tpath, "a", "a.go")}
	for _, scenario := range scenarios {
		// Instrument package "a", producing a set of instrumented output
		// files and an 'output config' file to pass on to the compiler.
		ppath := "cfg/a"
		pname := "a"
		mode := scenario.mode
		gran := scenario.gran
		tag := mode + "_" + gran
		incfg = writePkgConfig(t, instdira, tag, ppath, pname, gran, "")
		ofs, outcfg, _ := runPkgCover(t, instdira, tag, incfg, mode,
			apkgfiles, false)
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
		apkgfiles, errExpected)
	want := "error reading pkgconfig file"
	if !strings.Contains(errmsg, want) {
		t.Errorf("'bad config file' test: wanted %s got %s", want, errmsg)
	}

	// Expect err if config file contains unknown stuff.
	t.Logf("mangling in config")
	writeFile(t, incfg, []byte("blah=foo\n"))
	_, _, errmsg = runPkgCover(t, instdira, tag, incfg, mode,
		apkgfiles, errExpected)
	want = "error reading pkgconfig file"
	if !strings.Contains(errmsg, want) {
		t.Errorf("'bad config file' test: wanted %s got %s", want, errmsg)
	}

	// Expect error on empty config file.
	t.Logf("writing empty config")
	writeFile(t, incfg, []byte("\n"))
	_, _, errmsg = runPkgCover(t, instdira, tag, incfg, mode,
		apkgfiles, errExpected)
	if !strings.Contains(errmsg, want) {
		t.Errorf("'bad config file' test: wanted %s got %s", want, errmsg)
	}
}

func TestCoverOnPackageWithNoTestFiles(t *testing.T) {
	testenv.MustHaveGoRun(t)

	// For packages with no test files, the new "go test -cover"
	// strategy is to run cmd/cover on the package in a special
	// "EmitMetaFile" mode. When running in this mode, cmd/cover walks
	// the package doing instrumentation, but when finished, instead of
	// writing out instrumented source files, it directly emits a
	// meta-data file for the package in question, essentially
	// simulating the effect that you would get if you added a dummy
	// "no-op" x_test.go file and then did a build and run of the test.

	t.Run("YesFuncsNoTests", func(t *testing.T) {
		testCoverNoTestsYesFuncs(t)
	})
	t.Run("NoFuncsNoTests", func(t *testing.T) {
		testCoverNoTestsNoFuncs(t)
	})
}

func testCoverNoTestsYesFuncs(t *testing.T) {
	t.Parallel()
	dir := tempDir(t)

	// Run the cover command with "emit meta" enabled on a package
	// with functions but no test files.
	tpath := filepath.Join("testdata", "pkgcfg")
	pkg1files := []string{filepath.Join(tpath, "yesFuncsNoTests", "yfnt.go")}
	ppath := "cfg/yesFuncsNoTests"
	pname := "yesFuncsNoTests"
	mode := "count"
	gran := "perblock"
	tag := mode + "_" + gran
	instdir := filepath.Join(dir, "inst")
	if err := os.Mkdir(instdir, 0777); err != nil {
		t.Fatal(err)
	}
	mdir := filepath.Join(dir, "meta")
	if err := os.Mkdir(mdir, 0777); err != nil {
		t.Fatal(err)
	}
	mpath := filepath.Join(mdir, "covmeta.xxx")
	incfg := writePkgConfig(t, instdir, tag, ppath, pname, gran, mpath)
	_, _, errmsg := runPkgCover(t, instdir, tag, incfg, mode,
		pkg1files, false)
	if errmsg != "" {
		t.Fatalf("runPkgCover err: %q", errmsg)
	}

	// Check for existence of meta-data file.
	if inf, err := os.Open(mpath); err != nil {
		t.Fatalf("meta-data file not created: %v", err)
	} else {
		inf.Close()
	}

	// Make sure it is digestible.
	cdargs := []string{"tool", "covdata", "percent", "-i", mdir}
	cmd := testenv.Command(t, testenv.GoToolPath(t), cdargs...)
	run(cmd, t)
}

func testCoverNoTestsNoFuncs(t *testing.T) {
	t.Parallel()
	dir := tempDir(t)

	// Run the cover command with "emit meta" enabled on a package
	// with no functions and no test files.
	tpath := filepath.Join("testdata", "pkgcfg")
	pkgfiles := []string{filepath.Join(tpath, "noFuncsNoTests", "nfnt.go")}
	pname := "noFuncsNoTests"
	mode := "count"
	gran := "perblock"
	ppath := "cfg/" + pname
	tag := mode + "_" + gran
	instdir := filepath.Join(dir, "inst2")
	if err := os.Mkdir(instdir, 0777); err != nil {
		t.Fatal(err)
	}
	mdir := filepath.Join(dir, "meta2")
	if err := os.Mkdir(mdir, 0777); err != nil {
		t.Fatal(err)
	}
	mpath := filepath.Join(mdir, "covmeta.yyy")
	incfg := writePkgConfig(t, instdir, tag, ppath, pname, gran, mpath)
	_, _, errmsg := runPkgCover(t, instdir, tag, incfg, mode,
		pkgfiles, false)
	if errmsg != "" {
		t.Fatalf("runPkgCover err: %q", errmsg)
	}

	// We expect to see an empty meta-data file in this case.
	if inf, err := os.Open(mpath); err != nil {
		t.Fatalf("opening meta-data file: error %v", err)
	} else {
		defer inf.Close()
		fi, err := inf.Stat()
		if err != nil {
			t.Fatalf("stat meta-data file: %v", err)
		}
		if fi.Size() != 0 {
			t.Fatalf("want zero-sized meta-data file got size %d",
				fi.Size())
		}
	}
}
