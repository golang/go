// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go.1.16
// +build go.1.16

// Running this program in the tools directory will produce a coverage file /tmp/cover.out
// and a coverage report for all the packages under internal/lsp, accumulated by all the tests
// under gopls.
//
// -o controls where the coverage file is written, defaulting to /tmp/cover.out
// -i coverage-file will generate the report from an existing coverage file
// -v controls verbosity (0: only report coverage, 1: report as each directory is finished,
//
//	2: report on each test, 3: more details, 4: too much)
//
// -t tests only tests packages in the given comma-separated list of directories in gopls.
//
//	The names should start with ., as in ./internal/regtest/bench
//
// -run tests. If set, -run tests is passed on to the go test command.
//
// Despite gopls' use of goroutines, the counts are almost deterministic.
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"golang.org/x/tools/cover"
)

var (
	proFile = flag.String("i", "", "existing profile file")
	outFile = flag.String("o", "/tmp/cover.out", "where to write the coverage file")
	verbose = flag.Int("v", 0, "how much detail to print as tests are running")
	tests   = flag.String("t", "", "list of tests to run")
	run     = flag.String("run", "", "value of -run to pass to go test")
)

func main() {
	log.SetFlags(log.Lshortfile)
	flag.Parse()

	if *proFile != "" {
		report(*proFile)
		return
	}

	checkCwd()
	// find the packages under gopls containing tests
	tests := listDirs("gopls")
	tests = onlyTests(tests)
	tests = realTestName(tests)

	// report coverage for packages under internal/lsp
	parg := "golang.org/x/tools/gopls/internal/lsp/..."

	accum := []string{}
	seen := make(map[string]bool)
	now := time.Now()
	for _, toRun := range tests {
		if excluded(toRun) {
			continue
		}
		x := runTest(toRun, parg)
		if *verbose > 0 {
			fmt.Printf("finished %s %.1fs\n", toRun, time.Since(now).Seconds())
		}
		lines := bytes.Split(x, []byte{'\n'})
		for _, l := range lines {
			if len(l) == 0 {
				continue
			}
			if !seen[string(l)] {
				// not accumulating counts, so only works for mode:set
				seen[string(l)] = true
				accum = append(accum, string(l))
			}
		}
	}
	sort.Strings(accum[1:])
	if err := os.WriteFile(*outFile, []byte(strings.Join(accum, "\n")), 0644); err != nil {
		log.Print(err)
	}
	report(*outFile)
}

type result struct {
	Time    time.Time
	Test    string
	Action  string
	Package string
	Output  string
	Elapsed float64
}

func runTest(tName, parg string) []byte {
	args := []string{"test", "-short", "-coverpkg", parg, "-coverprofile", *outFile,
		"-json"}
	if *run != "" {
		args = append(args, fmt.Sprintf("-run=%s", *run))
	}
	args = append(args, tName)
	cmd := exec.Command("go", args...)
	cmd.Dir = "./gopls"
	ans, err := cmd.Output()
	if *verbose > 1 {
		got := strings.Split(string(ans), "\n")
		for _, g := range got {
			if g == "" {
				continue
			}
			var m result
			if err := json.Unmarshal([]byte(g), &m); err != nil {
				log.Printf("%T/%v", err, err) // shouldn't happen
				continue
			}
			maybePrint(m)
		}
	}
	if err != nil {
		log.Printf("%s: %q, cmd=%s", tName, ans, cmd.String())
	}
	buf, err := os.ReadFile(*outFile)
	if err != nil {
		log.Fatal(err)
	}
	return buf
}

func report(fn string) {
	profs, err := cover.ParseProfiles(fn)
	if err != nil {
		log.Fatal(err)
	}
	for _, p := range profs {
		statements, counts := 0, 0
		for _, x := range p.Blocks {
			statements += x.NumStmt
			if x.Count != 0 {
				counts += x.NumStmt // sic: if any were executed, all were
			}
		}
		pc := 100 * float64(counts) / float64(statements)
		fmt.Printf("%3.0f%% %3d/%3d %s\n", pc, counts, statements, p.FileName)
	}
}

var todo []string // tests to run

func excluded(tname string) bool {
	if *tests == "" { // run all tests
		return false
	}
	if todo == nil {
		todo = strings.Split(*tests, ",")
	}
	for _, nm := range todo {
		if tname == nm { // run this test
			return false
		}
	}
	// not in list, skip it
	return true
}

// should m.Package be printed sometime?
func maybePrint(m result) {
	switch m.Action {
	case "pass", "fail", "skip":
		fmt.Printf("%s %s %.3f\n", m.Action, m.Test, m.Elapsed)
	case "run":
		if *verbose > 2 {
			fmt.Printf("%s %s %.3f\n", m.Action, m.Test, m.Elapsed)
		}
	case "output":
		if *verbose > 3 {
			fmt.Printf("%s %s %q %.3f\n", m.Action, m.Test, m.Output, m.Elapsed)
		}
	case "pause", "cont":
		if *verbose > 2 {
			fmt.Printf("%s %s %.3f\n", m.Action, m.Test, m.Elapsed)
		}
	default:
		fmt.Printf("%#v\n", m)
		log.Fatalf("unknown action %s\n", m.Action)
	}
}

// return only the directories that contain tests
func onlyTests(s []string) []string {
	ans := []string{}
outer:
	for _, d := range s {
		files, err := os.ReadDir(d)
		if err != nil {
			log.Fatalf("%s: %v", d, err)
		}
		for _, de := range files {
			if strings.Contains(de.Name(), "_test.go") {
				ans = append(ans, d)
				continue outer
			}
		}
	}
	return ans
}

// replace the prefix gopls/ with ./ as the tests are run in the gopls directory
func realTestName(p []string) []string {
	ans := []string{}
	for _, x := range p {
		x = x[len("gopls/"):]
		ans = append(ans, "./"+x)
	}
	return ans
}

// make sure we start in a tools directory
func checkCwd() {
	dir, err := os.Getwd()
	if err != nil {
		log.Fatal(err)
	}
	// we expect to be at the root of golang.org/x/tools
	cmd := exec.Command("go", "list", "-m", "-f", "{{.Dir}}", "golang.org/x/tools")
	buf, err := cmd.Output()
	buf = bytes.Trim(buf, "\n \t") // remove \n at end
	if err != nil {
		log.Fatal(err)
	}
	if string(buf) != dir {
		log.Fatalf("wrong directory: in %q, should be in %q", dir, string(buf))
	}
	// and we expect gopls and internal/lsp as subdirectories
	_, err = os.Stat("gopls")
	if err != nil {
		log.Fatalf("expected a gopls directory, %v", err)
	}
}

func listDirs(dir string) []string {
	ans := []string{}
	f := func(path string, dirEntry os.DirEntry, err error) error {
		if strings.HasSuffix(path, "/testdata") || strings.HasSuffix(path, "/typescript") {
			return filepath.SkipDir
		}
		if dirEntry.IsDir() {
			ans = append(ans, path)
		}
		return nil
	}
	filepath.WalkDir(dir, f)
	return ans
}
