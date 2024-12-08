// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main_test

import (
	cmdcovdata "cmd/covdata"
	"flag"
	"fmt"
	"internal/coverage/pods"
	"internal/goexperiment"
	"internal/testenv"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"testing"
)

// Top level tempdir for test.
var testTempDir string

// If set, this will preserve all the tmpdir files from the test run.
var preserveTmp = flag.Bool("preservetmp", false, "keep tmpdir files for debugging")

// TestMain used here so that we can leverage the test executable
// itself as a cmd/covdata executable; compare to similar usage in
// the cmd/go tests.
func TestMain(m *testing.M) {
	// When CMDCOVDATA_TEST_RUN_MAIN is set, we're reusing the test
	// binary as cmd/cover. In this case we run the main func exported
	// via export_test.go, and exit; CMDCOVDATA_TEST_RUN_MAIN is set below
	// for actual test invocations.
	if os.Getenv("CMDCOVDATA_TEST_RUN_MAIN") != "" {
		cmdcovdata.Main()
		os.Exit(0)
	}
	flag.Parse()
	topTmpdir, err := os.MkdirTemp("", "cmd-covdata-test-")
	if err != nil {
		log.Fatal(err)
	}
	testTempDir = topTmpdir
	if !*preserveTmp {
		defer os.RemoveAll(topTmpdir)
	} else {
		fmt.Fprintf(os.Stderr, "debug: preserving tmpdir %s\n", topTmpdir)
	}
	os.Setenv("CMDCOVDATA_TEST_RUN_MAIN", "true")
	os.Exit(m.Run())
}

var tdmu sync.Mutex
var tdcount int

func tempDir(t *testing.T) string {
	tdmu.Lock()
	dir := filepath.Join(testTempDir, fmt.Sprintf("%03d", tdcount))
	tdcount++
	if err := os.Mkdir(dir, 0777); err != nil {
		t.Fatal(err)
	}
	defer tdmu.Unlock()
	return dir
}

const debugtrace = false

func gobuild(t *testing.T, indir string, bargs []string) {
	t.Helper()

	if debugtrace {
		if indir != "" {
			t.Logf("in dir %s: ", indir)
		}
		t.Logf("cmd: %s %+v\n", testenv.GoToolPath(t), bargs)
	}
	cmd := testenv.Command(t, testenv.GoToolPath(t), bargs...)
	cmd.Dir = indir
	b, err := cmd.CombinedOutput()
	if len(b) != 0 {
		t.Logf("## build output:\n%s", b)
	}
	if err != nil {
		t.Fatalf("build error: %v", err)
	}
}

func emitFile(t *testing.T, dst, src string) {
	payload, err := os.ReadFile(src)
	if err != nil {
		t.Fatalf("error reading %q: %v", src, err)
	}
	if err := os.WriteFile(dst, payload, 0666); err != nil {
		t.Fatalf("writing %q: %v", dst, err)
	}
}

const mainPkgPath = "prog"

func buildProg(t *testing.T, prog string, dir string, tag string, flags []string) (string, string) {
	// Create subdirs.
	subdir := filepath.Join(dir, prog+"dir"+tag)
	if err := os.Mkdir(subdir, 0777); err != nil {
		t.Fatalf("can't create outdir %s: %v", subdir, err)
	}
	depdir := filepath.Join(subdir, "dep")
	if err := os.Mkdir(depdir, 0777); err != nil {
		t.Fatalf("can't create outdir %s: %v", depdir, err)
	}

	// Emit program.
	insrc := filepath.Join("testdata", prog+".go")
	src := filepath.Join(subdir, prog+".go")
	emitFile(t, src, insrc)
	indep := filepath.Join("testdata", "dep.go")
	dep := filepath.Join(depdir, "dep.go")
	emitFile(t, dep, indep)

	// Emit go.mod.
	mod := filepath.Join(subdir, "go.mod")
	modsrc := "\nmodule " + mainPkgPath + "\n\ngo 1.19\n"
	if err := os.WriteFile(mod, []byte(modsrc), 0666); err != nil {
		t.Fatal(err)
	}
	exepath := filepath.Join(subdir, prog+".exe")
	bargs := []string{"build", "-cover", "-o", exepath}
	bargs = append(bargs, flags...)
	gobuild(t, subdir, bargs)
	return exepath, subdir
}

type state struct {
	dir      string
	exedir1  string
	exedir2  string
	exedir3  string
	exepath1 string
	exepath2 string
	exepath3 string
	tool     string
	outdirs  [4]string
}

const debugWorkDir = false

func TestCovTool(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	if !goexperiment.CoverageRedesign {
		t.Skipf("stubbed out due to goexperiment.CoverageRedesign=false")
	}
	dir := tempDir(t)
	if testing.Short() {
		t.Skip()
	}
	if debugWorkDir {
		// debugging
		dir = "/tmp/qqq"
		os.RemoveAll(dir)
		os.Mkdir(dir, 0777)
	}

	s := state{
		dir: dir,
	}
	s.exepath1, s.exedir1 = buildProg(t, "prog1", dir, "", nil)
	s.exepath2, s.exedir2 = buildProg(t, "prog2", dir, "", nil)
	flags := []string{"-covermode=atomic"}
	s.exepath3, s.exedir3 = buildProg(t, "prog1", dir, "atomic", flags)

	// Reuse unit test executable as tool to be tested.
	s.tool = testenv.Executable(t)

	// Create a few coverage output dirs.
	for i := 0; i < 4; i++ {
		d := filepath.Join(dir, fmt.Sprintf("covdata%d", i))
		s.outdirs[i] = d
		if err := os.Mkdir(d, 0777); err != nil {
			t.Fatalf("can't create outdir %s: %v", d, err)
		}
	}

	// Run instrumented program to generate some coverage data output files,
	// as follows:
	//
	//   <tmp>/covdata0   -- prog1.go compiled -cover
	//   <tmp>/covdata1   -- prog1.go compiled -cover
	//   <tmp>/covdata2   -- prog1.go compiled -covermode=atomic
	//   <tmp>/covdata3   -- prog1.go compiled -covermode=atomic
	//
	for m := 0; m < 2; m++ {
		for k := 0; k < 2; k++ {
			args := []string{}
			if k != 0 {
				args = append(args, "foo", "bar")
			}
			for i := 0; i <= k; i++ {
				exepath := s.exepath1
				if m != 0 {
					exepath = s.exepath3
				}
				cmd := testenv.Command(t, exepath, args...)
				cmd.Env = append(cmd.Env, "GOCOVERDIR="+s.outdirs[m*2+k])
				b, err := cmd.CombinedOutput()
				if len(b) != 0 {
					t.Logf("## instrumented run output:\n%s", b)
				}
				if err != nil {
					t.Fatalf("instrumented run error: %v", err)
				}
			}
		}
	}

	// At this point we can fork off a bunch of child tests
	// to check different tool modes.
	t.Run("MergeSimple", func(t *testing.T) {
		t.Parallel()
		testMergeSimple(t, s, s.outdirs[0], s.outdirs[1], "set")
		testMergeSimple(t, s, s.outdirs[2], s.outdirs[3], "atomic")
	})
	t.Run("MergeSelect", func(t *testing.T) {
		t.Parallel()
		testMergeSelect(t, s, s.outdirs[0], s.outdirs[1], "set")
		testMergeSelect(t, s, s.outdirs[2], s.outdirs[3], "atomic")
	})
	t.Run("MergePcombine", func(t *testing.T) {
		t.Parallel()
		testMergeCombinePrograms(t, s)
	})
	t.Run("Dump", func(t *testing.T) {
		t.Parallel()
		testDump(t, s)
	})
	t.Run("Percent", func(t *testing.T) {
		t.Parallel()
		testPercent(t, s)
	})
	t.Run("PkgList", func(t *testing.T) {
		t.Parallel()
		testPkgList(t, s)
	})
	t.Run("Textfmt", func(t *testing.T) {
		t.Parallel()
		testTextfmt(t, s)
	})
	t.Run("Subtract", func(t *testing.T) {
		t.Parallel()
		testSubtract(t, s)
	})
	t.Run("Intersect", func(t *testing.T) {
		t.Parallel()
		testIntersect(t, s, s.outdirs[0], s.outdirs[1], "set")
		testIntersect(t, s, s.outdirs[2], s.outdirs[3], "atomic")
	})
	t.Run("CounterClash", func(t *testing.T) {
		t.Parallel()
		testCounterClash(t, s)
	})
	t.Run("TestEmpty", func(t *testing.T) {
		t.Parallel()
		testEmpty(t, s)
	})
	t.Run("TestCommandLineErrors", func(t *testing.T) {
		t.Parallel()
		testCommandLineErrors(t, s, s.outdirs[0])
	})
}

const showToolInvocations = true

func runToolOp(t *testing.T, s state, op string, args []string) []string {
	// Perform tool run.
	t.Helper()
	args = append([]string{op}, args...)
	if showToolInvocations {
		t.Logf("%s cmd is: %s %+v", op, s.tool, args)
	}
	cmd := testenv.Command(t, s.tool, args...)
	b, err := cmd.CombinedOutput()
	if err != nil {
		fmt.Fprintf(os.Stderr, "## %s output: %s\n", op, b)
		t.Fatalf("%q run error: %v", op, err)
	}
	output := strings.TrimSpace(string(b))
	lines := strings.Split(output, "\n")
	if len(lines) == 1 && lines[0] == "" {
		lines = nil
	}
	return lines
}

func testDump(t *testing.T, s state) {
	// Run the dumper on the two dirs we generated.
	dargs := []string{"-pkg=" + mainPkgPath, "-live", "-i=" + s.outdirs[0] + "," + s.outdirs[1]}
	lines := runToolOp(t, s, "debugdump", dargs)

	// Sift through the output to make sure it has some key elements.
	testpoints := []struct {
		tag string
		re  *regexp.Regexp
	}{
		{
			"args",
			regexp.MustCompile(`^data file .+ GOOS=.+ GOARCH=.+ program args: .+$`),
		},
		{
			"main package",
			regexp.MustCompile(`^Package path: ` + mainPkgPath + `\s*$`),
		},
		{
			"main function",
			regexp.MustCompile(`^Func: main\s*$`),
		},
	}

	bad := false
	for _, testpoint := range testpoints {
		found := false
		for _, line := range lines {
			if m := testpoint.re.FindStringSubmatch(line); m != nil {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("dump output regexp match failed for %q", testpoint.tag)
			bad = true
		}
	}
	if bad {
		dumplines(lines)
	}
}

func testPercent(t *testing.T, s state) {
	// Run the dumper on the two dirs we generated.
	dargs := []string{"-pkg=" + mainPkgPath, "-i=" + s.outdirs[0] + "," + s.outdirs[1]}
	lines := runToolOp(t, s, "percent", dargs)

	// Sift through the output to make sure it has the needful.
	testpoints := []struct {
		tag string
		re  *regexp.Regexp
	}{
		{
			"statement coverage percent",
			regexp.MustCompile(`coverage: \d+\.\d% of statements\s*$`),
		},
	}

	bad := false
	for _, testpoint := range testpoints {
		found := false
		for _, line := range lines {
			if m := testpoint.re.FindStringSubmatch(line); m != nil {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("percent output regexp match failed for %s", testpoint.tag)
			bad = true
		}
	}
	if bad {
		dumplines(lines)
	}
}

func testPkgList(t *testing.T, s state) {
	dargs := []string{"-i=" + s.outdirs[0] + "," + s.outdirs[1]}
	lines := runToolOp(t, s, "pkglist", dargs)

	want := []string{mainPkgPath, mainPkgPath + "/dep"}
	bad := false
	if len(lines) != 2 {
		t.Errorf("expect pkglist to return two lines")
		bad = true
	} else {
		for i := 0; i < 2; i++ {
			lines[i] = strings.TrimSpace(lines[i])
			if want[i] != lines[i] {
				t.Errorf("line %d want %s got %s", i, want[i], lines[i])
				bad = true
			}
		}
	}
	if bad {
		dumplines(lines)
	}
}

func testTextfmt(t *testing.T, s state) {
	outf := s.dir + "/" + "t.txt"
	dargs := []string{"-pkg=" + mainPkgPath, "-i=" + s.outdirs[0] + "," + s.outdirs[1],
		"-o", outf}
	lines := runToolOp(t, s, "textfmt", dargs)

	// No output expected.
	if len(lines) != 0 {
		dumplines(lines)
		t.Errorf("unexpected output from go tool covdata textfmt")
	}

	// Open and read the first few bits of the file.
	payload, err := os.ReadFile(outf)
	if err != nil {
		t.Errorf("opening %s: %v\n", outf, err)
	}
	lines = strings.Split(string(payload), "\n")
	want0 := "mode: set"
	if lines[0] != want0 {
		dumplines(lines[0:10])
		t.Errorf("textfmt: want %s got %s", want0, lines[0])
	}
	want1 := mainPkgPath + "/prog1.go:13.14,15.2 1 1"
	if lines[1] != want1 {
		dumplines(lines[0:10])
		t.Errorf("textfmt: want %s got %s", want1, lines[1])
	}
}

func dumplines(lines []string) {
	for i := range lines {
		fmt.Fprintf(os.Stderr, "%s\n", lines[i])
	}
}

type dumpCheck struct {
	tag     string
	re      *regexp.Regexp
	negate  bool
	nonzero bool
	zero    bool
}

// runDumpChecks examines the output of "go tool covdata debugdump"
// for a given output directory, looking for the presence or absence
// of specific markers.
func runDumpChecks(t *testing.T, s state, dir string, flags []string, checks []dumpCheck) {
	dargs := []string{"-i", dir}
	dargs = append(dargs, flags...)
	lines := runToolOp(t, s, "debugdump", dargs)
	if len(lines) == 0 {
		t.Fatalf("dump run produced no output")
	}

	bad := false
	for _, check := range checks {
		found := false
		for _, line := range lines {
			if m := check.re.FindStringSubmatch(line); m != nil {
				found = true
				if check.negate {
					t.Errorf("tag %q: unexpected match", check.tag)
					bad = true

				}
				if check.nonzero || check.zero {
					if len(m) < 2 {
						t.Errorf("tag %s: submatch failed (short m)", check.tag)
						bad = true
						continue
					}
					if m[1] == "" {
						t.Errorf("tag %s: submatch failed", check.tag)
						bad = true
						continue
					}
					i, err := strconv.Atoi(m[1])
					if err != nil {
						t.Errorf("tag %s: match Atoi failed on %s",
							check.tag, m[1])
						continue
					}
					if check.zero && i != 0 {
						t.Errorf("tag %s: match zero failed on %s",
							check.tag, m[1])
					} else if check.nonzero && i == 0 {
						t.Errorf("tag %s: match nonzero failed on %s",
							check.tag, m[1])
					}
				}
				break
			}
		}
		if !found && !check.negate {
			t.Errorf("dump output regexp match failed for %s", check.tag)
			bad = true
		}
	}
	if bad {
		fmt.Printf("output from 'dump' run:\n")
		dumplines(lines)
	}
}

func testMergeSimple(t *testing.T, s state, indir1, indir2, tag string) {
	outdir := filepath.Join(s.dir, "simpleMergeOut"+tag)
	if err := os.Mkdir(outdir, 0777); err != nil {
		t.Fatalf("can't create outdir %s: %v", outdir, err)
	}

	// Merge the two dirs into a final result.
	ins := fmt.Sprintf("-i=%s,%s", indir1, indir2)
	out := fmt.Sprintf("-o=%s", outdir)
	margs := []string{ins, out}
	lines := runToolOp(t, s, "merge", margs)
	if len(lines) != 0 {
		t.Errorf("merge run produced %d lines of unexpected output", len(lines))
		dumplines(lines)
	}

	// We expect the merge tool to produce exactly two files: a meta
	// data file and a counter file. If we get more than just this one
	// pair, something went wrong.
	podlist, err := pods.CollectPods([]string{outdir}, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(podlist) != 1 {
		t.Fatalf("expected 1 pod, got %d pods", len(podlist))
	}
	ncdfs := len(podlist[0].CounterDataFiles)
	if ncdfs != 1 {
		t.Fatalf("expected 1 counter data file, got %d", ncdfs)
	}

	// Sift through the output to make sure it has some key elements.
	// In particular, we want to see entries for all three functions
	// ("first", "second", and "third").
	testpoints := []dumpCheck{
		{
			tag: "first function",
			re:  regexp.MustCompile(`^Func: first\s*$`),
		},
		{
			tag: "second function",
			re:  regexp.MustCompile(`^Func: second\s*$`),
		},
		{
			tag: "third function",
			re:  regexp.MustCompile(`^Func: third\s*$`),
		},
		{
			tag:     "third function unit 0",
			re:      regexp.MustCompile(`^0: L23:C23 -- L24:C12 NS=1 = (\d+)$`),
			nonzero: true,
		},
		{
			tag:     "third function unit 1",
			re:      regexp.MustCompile(`^1: L27:C2 -- L28:C10 NS=2 = (\d+)$`),
			nonzero: true,
		},
		{
			tag:     "third function unit 2",
			re:      regexp.MustCompile(`^2: L24:C12 -- L26:C3 NS=1 = (\d+)$`),
			nonzero: true,
		},
	}
	flags := []string{"-live", "-pkg=" + mainPkgPath}
	runDumpChecks(t, s, outdir, flags, testpoints)
}

func testMergeSelect(t *testing.T, s state, indir1, indir2 string, tag string) {
	outdir := filepath.Join(s.dir, "selectMergeOut"+tag)
	if err := os.Mkdir(outdir, 0777); err != nil {
		t.Fatalf("can't create outdir %s: %v", outdir, err)
	}

	// Merge two input dirs into a final result, but filter
	// based on package.
	ins := fmt.Sprintf("-i=%s,%s", indir1, indir2)
	out := fmt.Sprintf("-o=%s", outdir)
	margs := []string{"-pkg=" + mainPkgPath + "/dep", ins, out}
	lines := runToolOp(t, s, "merge", margs)
	if len(lines) != 0 {
		t.Errorf("merge run produced %d lines of unexpected output", len(lines))
		dumplines(lines)
	}

	// Dump the files in the merged output dir and examine the result.
	// We expect to see only the functions in package "dep".
	dargs := []string{"-i=" + outdir}
	lines = runToolOp(t, s, "debugdump", dargs)
	if len(lines) == 0 {
		t.Fatalf("dump run produced no output")
	}
	want := map[string]int{
		"Package path: " + mainPkgPath + "/dep": 0,
		"Func: Dep1":                            0,
		"Func: PDep":                            0,
	}
	bad := false
	for _, line := range lines {
		if v, ok := want[line]; ok {
			if v != 0 {
				t.Errorf("duplicate line %s", line)
				bad = true
				break
			}
			want[line] = 1
			continue
		}
		// no other functions or packages expected.
		if strings.HasPrefix(line, "Func:") || strings.HasPrefix(line, "Package path:") {
			t.Errorf("unexpected line: %s", line)
			bad = true
			break
		}
	}
	if bad {
		dumplines(lines)
	}
}

func testMergeCombinePrograms(t *testing.T, s state) {

	// Run the new program, emitting output into a new set
	// of outdirs.
	runout := [2]string{}
	for k := 0; k < 2; k++ {
		runout[k] = filepath.Join(s.dir, fmt.Sprintf("newcovdata%d", k))
		if err := os.Mkdir(runout[k], 0777); err != nil {
			t.Fatalf("can't create outdir %s: %v", runout[k], err)
		}
		args := []string{}
		if k != 0 {
			args = append(args, "foo", "bar")
		}
		cmd := testenv.Command(t, s.exepath2, args...)
		cmd.Env = append(cmd.Env, "GOCOVERDIR="+runout[k])
		b, err := cmd.CombinedOutput()
		if len(b) != 0 {
			t.Logf("## instrumented run output:\n%s", b)
		}
		if err != nil {
			t.Fatalf("instrumented run error: %v", err)
		}
	}

	// Create out dir for -pcombine merge.
	moutdir := filepath.Join(s.dir, "mergeCombineOut")
	if err := os.Mkdir(moutdir, 0777); err != nil {
		t.Fatalf("can't create outdir %s: %v", moutdir, err)
	}

	// Run a merge over both programs, using the -pcombine
	// flag to do maximal combining.
	ins := fmt.Sprintf("-i=%s,%s,%s,%s", s.outdirs[0], s.outdirs[1],
		runout[0], runout[1])
	out := fmt.Sprintf("-o=%s", moutdir)
	margs := []string{"-pcombine", ins, out}
	lines := runToolOp(t, s, "merge", margs)
	if len(lines) != 0 {
		t.Errorf("merge run produced unexpected output: %v", lines)
	}

	// We expect the merge tool to produce exactly two files: a meta
	// data file and a counter file. If we get more than just this one
	// pair, something went wrong.
	podlist, err := pods.CollectPods([]string{moutdir}, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(podlist) != 1 {
		t.Fatalf("expected 1 pod, got %d pods", len(podlist))
	}
	ncdfs := len(podlist[0].CounterDataFiles)
	if ncdfs != 1 {
		t.Fatalf("expected 1 counter data file, got %d", ncdfs)
	}

	// Sift through the output to make sure it has some key elements.
	testpoints := []dumpCheck{
		{
			tag: "first function",
			re:  regexp.MustCompile(`^Func: first\s*$`),
		},
		{
			tag: "sixth function",
			re:  regexp.MustCompile(`^Func: sixth\s*$`),
		},
	}

	flags := []string{"-live", "-pkg=" + mainPkgPath}
	runDumpChecks(t, s, moutdir, flags, testpoints)
}

func testSubtract(t *testing.T, s state) {
	// Create out dir for subtract merge.
	soutdir := filepath.Join(s.dir, "subtractOut")
	if err := os.Mkdir(soutdir, 0777); err != nil {
		t.Fatalf("can't create outdir %s: %v", soutdir, err)
	}

	// Subtract the two dirs into a final result.
	ins := fmt.Sprintf("-i=%s,%s", s.outdirs[0], s.outdirs[1])
	out := fmt.Sprintf("-o=%s", soutdir)
	sargs := []string{ins, out}
	lines := runToolOp(t, s, "subtract", sargs)
	if len(lines) != 0 {
		t.Errorf("subtract run produced unexpected output: %+v", lines)
	}

	// Dump the files in the subtract output dir and examine the result.
	dargs := []string{"-pkg=" + mainPkgPath, "-live", "-i=" + soutdir}
	lines = runToolOp(t, s, "debugdump", dargs)
	if len(lines) == 0 {
		t.Errorf("dump run produced no output")
	}

	// Vet the output.
	testpoints := []dumpCheck{
		{
			tag: "first function",
			re:  regexp.MustCompile(`^Func: first\s*$`),
		},
		{
			tag: "dep function",
			re:  regexp.MustCompile(`^Func: Dep1\s*$`),
		},
		{
			tag: "third function",
			re:  regexp.MustCompile(`^Func: third\s*$`),
		},
		{
			tag:  "third function unit 0",
			re:   regexp.MustCompile(`^0: L23:C23 -- L24:C12 NS=1 = (\d+)$`),
			zero: true,
		},
		{
			tag:     "third function unit 1",
			re:      regexp.MustCompile(`^1: L27:C2 -- L28:C10 NS=2 = (\d+)$`),
			nonzero: true,
		},
		{
			tag:  "third function unit 2",
			re:   regexp.MustCompile(`^2: L24:C12 -- L26:C3 NS=1 = (\d+)$`),
			zero: true,
		},
	}
	flags := []string{}
	runDumpChecks(t, s, soutdir, flags, testpoints)
}

func testIntersect(t *testing.T, s state, indir1, indir2, tag string) {
	// Create out dir for intersection.
	ioutdir := filepath.Join(s.dir, "intersectOut"+tag)
	if err := os.Mkdir(ioutdir, 0777); err != nil {
		t.Fatalf("can't create outdir %s: %v", ioutdir, err)
	}

	// Intersect the two dirs into a final result.
	ins := fmt.Sprintf("-i=%s,%s", indir1, indir2)
	out := fmt.Sprintf("-o=%s", ioutdir)
	sargs := []string{ins, out}
	lines := runToolOp(t, s, "intersect", sargs)
	if len(lines) != 0 {
		t.Errorf("intersect run produced unexpected output: %+v", lines)
	}

	// Dump the files in the subtract output dir and examine the result.
	dargs := []string{"-pkg=" + mainPkgPath, "-live", "-i=" + ioutdir}
	lines = runToolOp(t, s, "debugdump", dargs)
	if len(lines) == 0 {
		t.Errorf("dump run produced no output")
	}

	// Vet the output.
	testpoints := []dumpCheck{
		{
			tag:    "first function",
			re:     regexp.MustCompile(`^Func: first\s*$`),
			negate: true,
		},
		{
			tag: "third function",
			re:  regexp.MustCompile(`^Func: third\s*$`),
		},
	}
	flags := []string{"-live"}
	runDumpChecks(t, s, ioutdir, flags, testpoints)
}

func testCounterClash(t *testing.T, s state) {
	// Create out dir.
	ccoutdir := filepath.Join(s.dir, "ccOut")
	if err := os.Mkdir(ccoutdir, 0777); err != nil {
		t.Fatalf("can't create outdir %s: %v", ccoutdir, err)
	}

	// Try to merge covdata0 (from prog1.go -countermode=set) with
	// covdata1 (from prog1.go -countermode=atomic"). This should
	// work properly, but result in multiple meta-data files.
	ins := fmt.Sprintf("-i=%s,%s", s.outdirs[0], s.outdirs[3])
	out := fmt.Sprintf("-o=%s", ccoutdir)
	args := append([]string{}, "merge", ins, out, "-pcombine")
	if debugtrace {
		t.Logf("cc merge command is %s %v\n", s.tool, args)
	}
	cmd := testenv.Command(t, s.tool, args...)
	b, err := cmd.CombinedOutput()
	t.Logf("%% output: %s\n", string(b))
	if err != nil {
		t.Fatalf("clash merge failed: %v", err)
	}

	// Ask for a textual report from the two dirs. Here we have
	// to report the mode clash.
	out = "-o=" + filepath.Join(ccoutdir, "file.txt")
	args = append([]string{}, "textfmt", ins, out)
	if debugtrace {
		t.Logf("clash textfmt command is %s %v\n", s.tool, args)
	}
	cmd = testenv.Command(t, s.tool, args...)
	b, err = cmd.CombinedOutput()
	t.Logf("%% output: %s\n", string(b))
	if err == nil {
		t.Fatalf("expected mode clash")
	}
	got := string(b)
	want := "counter mode clash while reading meta-data"
	if !strings.Contains(got, want) {
		t.Errorf("counter clash textfmt: wanted %s got %s", want, got)
	}
}

func testEmpty(t *testing.T, s state) {

	// Create a new empty directory.
	empty := filepath.Join(s.dir, "empty")
	if err := os.Mkdir(empty, 0777); err != nil {
		t.Fatalf("can't create dir %s: %v", empty, err)
	}

	// Create out dir.
	eoutdir := filepath.Join(s.dir, "emptyOut")
	if err := os.Mkdir(eoutdir, 0777); err != nil {
		t.Fatalf("can't create outdir %s: %v", eoutdir, err)
	}

	// Run various operations (merge, dump, textfmt, and so on)
	// using the empty directory. We're not interested in the output
	// here, just making sure that you can do these runs without
	// any error or crash.

	scenarios := []struct {
		tag  string
		args []string
	}{
		{
			tag:  "merge",
			args: []string{"merge", "-o", eoutdir},
		},
		{
			tag:  "textfmt",
			args: []string{"textfmt", "-o", filepath.Join(eoutdir, "foo.txt")},
		},
		{
			tag:  "func",
			args: []string{"func"},
		},
		{
			tag:  "pkglist",
			args: []string{"pkglist"},
		},
		{
			tag:  "debugdump",
			args: []string{"debugdump"},
		},
		{
			tag:  "percent",
			args: []string{"percent"},
		},
	}

	for _, x := range scenarios {
		ins := fmt.Sprintf("-i=%s", empty)
		args := append([]string{}, x.args...)
		args = append(args, ins)
		if false {
			t.Logf("cmd is %s %v\n", s.tool, args)
		}
		cmd := testenv.Command(t, s.tool, args...)
		b, err := cmd.CombinedOutput()
		t.Logf("%% output: %s\n", string(b))
		if err != nil {
			t.Fatalf("command %s %+v failed with %v",
				s.tool, x.args, err)
		}
	}
}

func testCommandLineErrors(t *testing.T, s state, outdir string) {

	// Create out dir.
	eoutdir := filepath.Join(s.dir, "errorsOut")
	if err := os.Mkdir(eoutdir, 0777); err != nil {
		t.Fatalf("can't create outdir %s: %v", eoutdir, err)
	}

	// Run various operations (merge, dump, textfmt, and so on)
	// using the empty directory. We're not interested in the output
	// here, just making sure that you can do these runs without
	// any error or crash.

	scenarios := []struct {
		tag  string
		args []string
		exp  string
	}{
		{
			tag:  "input missing",
			args: []string{"merge", "-o", eoutdir, "-i", "not there"},
			exp:  "error: reading inputs: ",
		},
		{
			tag:  "badv",
			args: []string{"textfmt", "-i", outdir, "-v=abc"},
		},
	}

	for _, x := range scenarios {
		args := append([]string{}, x.args...)
		if false {
			t.Logf("cmd is %s %v\n", s.tool, args)
		}
		cmd := testenv.Command(t, s.tool, args...)
		b, err := cmd.CombinedOutput()
		if err == nil {
			t.Logf("%% output: %s\n", string(b))
			t.Fatalf("command %s %+v unexpectedly succeeded",
				s.tool, x.args)
		} else {
			if !strings.Contains(string(b), x.exp) {
				t.Fatalf("command %s %+v:\ngot:\n%s\nwanted to see: %v\n",
					s.tool, x.args, string(b), x.exp)
			}
		}
	}
}
