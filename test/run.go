// #ignore

// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Run runs tests in the test directory.
// 
// TODO(bradfitz): docs of some sort, once we figure out how we're changing
// headers of files
package main

import (
	"bytes"
	"errors"
	"flag"
	"fmt"
	"go/build"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
)

var (
	verbose     = flag.Bool("v", false, "verbose. if set, parallelism is set to 1.")
	numParallel = flag.Int("n", runtime.NumCPU(), "number of parallel tests to run")
	summary     = flag.Bool("summary", false, "show summary of results")
	showSkips   = flag.Bool("show_skips", false, "show skipped tests")
)

var (
	// gc and ld are [568][gl].
	gc, ld string

	// letter is the build.ArchChar
	letter string

	// dirs are the directories to look for *.go files in.
	// TODO(bradfitz): just use all directories?
	dirs = []string{".", "ken", "chan", "interface", "syntax", "dwarf", "fixedbugs", "bugs"}

	// ratec controls the max number of tests running at a time.
	ratec chan bool

	// toRun is the channel of tests to run.
	// It is nil until the first test is started.
	toRun chan *test
)

// maxTests is an upper bound on the total number of tests.
// It is used as a channel buffer size to make sure sends don't block.
const maxTests = 5000

func main() {
	flag.Parse()

	// Disable parallelism if printing
	if *verbose {
		*numParallel = 1
	}

	ratec = make(chan bool, *numParallel)
	var err error
	letter, err = build.ArchChar(build.Default.GOARCH)
	check(err)
	gc = letter + "g"
	ld = letter + "l"

	var tests []*test
	if flag.NArg() > 0 {
		for _, arg := range flag.Args() {
			if arg == "-" || arg == "--" {
				// Permit running either:
				// $ go run run.go - env.go
				// $ go run run.go -- env.go
				continue
			}
			if !strings.HasSuffix(arg, ".go") {
				log.Fatalf("can't yet deal with non-go file %q", arg)
			}
			dir, file := filepath.Split(arg)
			tests = append(tests, startTest(dir, file))
		}
	} else {
		for _, dir := range dirs {
			for _, baseGoFile := range goFiles(dir) {
				tests = append(tests, startTest(dir, baseGoFile))
			}
		}
	}

	failed := false
	resCount := map[string]int{}
	for _, test := range tests {
		<-test.donec
		_, isSkip := test.err.(skipError)
		errStr := "pass"
		if isSkip {
			errStr = "skip"
		}
		if test.err != nil {
			errStr = test.err.Error()
			if !isSkip {
				failed = true
			}
		}
		resCount[errStr]++
		if isSkip && !*verbose && !*showSkips {
			continue
		}
		if !*verbose && test.err == nil {
			continue
		}
		fmt.Printf("%-10s %-20s: %s\n", test.action, test.goFileName(), errStr)
	}

	if *summary {
		for k, v := range resCount {
			fmt.Printf("%5d %s\n", v, k)
		}
	}

	if failed {
		os.Exit(1)
	}
}

func toolPath(name string) string {
	p := filepath.Join(os.Getenv("GOROOT"), "bin", "tool", name)
	if _, err := os.Stat(p); err != nil {
		log.Fatalf("didn't find binary at %s", p)
	}
	return p
}

func goFiles(dir string) []string {
	f, err := os.Open(dir)
	check(err)
	dirnames, err := f.Readdirnames(-1)
	check(err)
	names := []string{}
	for _, name := range dirnames {
		if strings.HasSuffix(name, ".go") {
			names = append(names, name)
		}
	}
	sort.Strings(names)
	return names
}

// skipError describes why a test was skipped.
type skipError string

func (s skipError) Error() string { return string(s) }

func check(err error) {
	if err != nil {
		log.Fatal(err)
	}
}

// test holds the state of a test.
type test struct {
	dir, gofile string
	donec       chan bool // closed when done

	src    string
	action string // "compile", "build", "run", "errorcheck"

	tempDir string
	err     error
}

// startTest 
func startTest(dir, gofile string) *test {
	t := &test{
		dir:    dir,
		gofile: gofile,
		donec:  make(chan bool, 1),
	}
	if toRun == nil {
		toRun = make(chan *test, maxTests)
		go runTests()
	}
	select {
	case toRun <- t:
	default:
		panic("toRun buffer size (maxTests) is too small")
	}
	return t
}

// runTests runs tests in parallel, but respecting the order they
// were enqueued on the toRun channel.
func runTests() {
	for {
		ratec <- true
		t := <-toRun
		go func() {
			t.run()
			<-ratec
		}()
	}
}

var cwd, _ = os.Getwd()

func (t *test) goFileName() string {
	return filepath.Join(t.dir, t.gofile)
}

// run runs a test.
func (t *test) run() {
	defer close(t.donec)

	srcBytes, err := ioutil.ReadFile(t.goFileName())
	if err != nil {
		t.err = err
		return
	}
	t.src = string(srcBytes)
	if t.src[0] == '\n' {
		t.err = skipError("starts with newline")
		return
	}
	pos := strings.Index(t.src, "\n\n")
	if pos == -1 {
		t.err = errors.New("double newline not found")
		return
	}
	action := t.src[:pos]
	if strings.HasPrefix(action, "//") {
		action = action[2:]
	}
	
	var args []string
	f := strings.Fields(action)
	if len(f) > 0 {
		action = f[0]
		args = f[1:]
	}

	switch action {
	case "cmpout":
		action = "run" // the run case already looks for <dir>/<test>.out files
		fallthrough
	case "compile", "build", "run", "errorcheck":
		t.action = action
	default:
		t.err = skipError("skipped; unknown pattern: " + action)
		t.action = "??"
		return
	}

	t.makeTempDir()
	defer os.RemoveAll(t.tempDir)

	err = ioutil.WriteFile(filepath.Join(t.tempDir, t.gofile), srcBytes, 0644)
	check(err)
	
	useTmp := true
	runcmd := func(args ...string) ([]byte, error) {
		cmd := exec.Command(args[0], args[1:]...)
		var buf bytes.Buffer
		cmd.Stdout = &buf
		cmd.Stderr = &buf
		if useTmp {
			cmd.Dir = t.tempDir
		}
		cmd.Env = append(cmd.Env, "GOOS="+runtime.GOOS, "GOARCH="+runtime.GOARCH)
		err := cmd.Run()
		return buf.Bytes(), err
	}

	long := filepath.Join(cwd, t.goFileName())
	switch action {	
	default:
		t.err = fmt.Errorf("unimplemented action %q", action)

	case "errorcheck":
		out, _ := runcmd("go", "tool", gc, "-e", "-o", "a."+letter, long)
		t.err = t.errorCheck(string(out), long, t.gofile)
		return
	
	case "compile":
		out, err := runcmd("go", "tool", gc, "-e", "-o", "a."+letter, long)
		if err != nil {
			t.err = fmt.Errorf("%s\n%s", err, out)
		}
	
	case "build":
		out, err := runcmd("go", "build", "-o", "a.exe", long)
		if err != nil {
			t.err = fmt.Errorf("%s\n%s", err, out)
		}
	
	case "run":
		useTmp = false
		out, err := runcmd(append([]string{"go", "run", t.goFileName()}, args...)...)
		if err != nil {
			t.err = fmt.Errorf("%s\n%s", err, out)
		}
		if string(out) != t.expectedOutput() {
			t.err = fmt.Errorf("incorrect output\n%s", out)
		}
	}
}

func (t *test) String() string {
	return filepath.Join(t.dir, t.gofile)
}

func (t *test) makeTempDir() {
	var err error
	t.tempDir, err = ioutil.TempDir("", "")
	check(err)
}

func (t *test) expectedOutput() string {
	filename := filepath.Join(t.dir, t.gofile)
	filename = filename[:len(filename)-len(".go")]
	filename += ".out"
	b, _ := ioutil.ReadFile(filename)
	return string(b)
}

func (t *test) errorCheck(outStr string, full, short string) (err error) {
	defer func() {
		if *verbose && err != nil {
			log.Printf("%s gc output:\n%s", t, outStr)
		}
	}()
	var errs []error

	var out []string
	// 6g error messages continue onto additional lines with leading tabs.
	// Split the output at the beginning of each line that doesn't begin with a tab.
	for _, line := range strings.Split(outStr, "\n") {
		if strings.HasPrefix(line, "\t") {
			out[len(out)-1] += "\n" + line
		} else {
			out = append(out, line)
		}
	}

	// Cut directory name.
	for i := range out {
		out[i] = strings.Replace(out[i], full, short, -1)
	}

	for _, we := range t.wantedErrors() {
		var errmsgs []string
		errmsgs, out = partitionStrings(we.filterRe, out)
		if len(errmsgs) == 0 {
			errs = append(errs, fmt.Errorf("%s:%d: missing error %q", we.file, we.lineNum, we.reStr))
			continue
		}
		matched := false
		for _, errmsg := range errmsgs {
			if we.re.MatchString(errmsg) {
				matched = true
			} else {
				out = append(out, errmsg)
			}
		}
		if !matched {
			errs = append(errs, fmt.Errorf("%s:%d: no match for %q in%s", we.file, we.lineNum, we.reStr, strings.Join(out, "\n")))
			continue
		}
	}

	if len(errs) == 0 {
		return nil
	}
	if len(errs) == 1 {
		return errs[0]
	}
	var buf bytes.Buffer
	fmt.Fprintf(&buf, "\n")
	for _, err := range errs {
		fmt.Fprintf(&buf, "%s\n", err.Error())
	}
	return errors.New(buf.String())

}

func partitionStrings(rx *regexp.Regexp, strs []string) (matched, unmatched []string) {
	for _, s := range strs {
		if rx.MatchString(s) {
			matched = append(matched, s)
		} else {
			unmatched = append(unmatched, s)
		}
	}
	return
}

type wantedError struct {
	reStr    string
	re       *regexp.Regexp
	lineNum  int
	file     string
	filterRe *regexp.Regexp // /^file:linenum\b/m
}

var (
	errRx       = regexp.MustCompile(`// (?:GC_)?ERROR (.*)`)
	errQuotesRx = regexp.MustCompile(`"([^"]*)"`)
	lineRx      = regexp.MustCompile(`LINE(([+-])([0-9]+))?`)
)

func (t *test) wantedErrors() (errs []wantedError) {
	for i, line := range strings.Split(t.src, "\n") {
		lineNum := i + 1
		if strings.Contains(line, "////") {
			// double comment disables ERROR
			continue
		}
		m := errRx.FindStringSubmatch(line)
		if m == nil {
			continue
		}
		all := m[1]
		mm := errQuotesRx.FindAllStringSubmatch(all, -1)
		if mm == nil {
			log.Fatalf("invalid errchk line in %s: %s", t.goFileName(), line)
		}
		for _, m := range mm {
			rx := lineRx.ReplaceAllStringFunc(m[1], func(m string) string {
				n := lineNum
				if strings.HasPrefix(m, "LINE+") {
					delta, _ := strconv.Atoi(m[5:])
					n += delta
				} else if strings.HasPrefix(m, "LINE-") {
					delta, _ := strconv.Atoi(m[5:])
					n -= delta
				}
				return fmt.Sprintf("%s:%d", t.gofile, n)
			})
			filterPattern := fmt.Sprintf(`^(\w+/)?%s:%d[:[]`, t.gofile, lineNum)
			errs = append(errs, wantedError{
				reStr:    rx,
				re:       regexp.MustCompile(rx),
				filterRe: regexp.MustCompile(filterPattern),
				lineNum:  lineNum,
				file:     t.gofile,
			})
		}
	}

	return
}
