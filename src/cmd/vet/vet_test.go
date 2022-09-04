// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main_test

import (
	"bytes"
	"errors"
	"fmt"
	"internal/testenv"
	"log"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"testing"
)

const dataDir = "testdata"

var binary string

// We implement TestMain so remove the test binary when all is done.
func TestMain(m *testing.M) {
	os.Exit(testMain(m))
}

func testMain(m *testing.M) int {
	dir, err := os.MkdirTemp("", "vet_test")
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	defer os.RemoveAll(dir)
	binary = filepath.Join(dir, "testvet.exe")

	return m.Run()
}

var (
	buildMu sync.Mutex // guards following
	built   = false    // We have built the binary.
	failed  = false    // We have failed to build the binary, don't try again.
)

func Build(t *testing.T) {
	buildMu.Lock()
	defer buildMu.Unlock()
	if built {
		return
	}
	if failed {
		t.Skip("cannot run on this environment")
	}
	testenv.MustHaveGoBuild(t)
	cmd := exec.Command(testenv.GoToolPath(t), "build", "-o", binary)
	output, err := cmd.CombinedOutput()
	if err != nil {
		failed = true
		fmt.Fprintf(os.Stderr, "%s\n", output)
		t.Fatal(err)
	}
	built = true
}

func vetCmd(t *testing.T, arg, pkg string) *exec.Cmd {
	cmd := exec.Command(testenv.GoToolPath(t), "vet", "-vettool="+binary, arg, path.Join("cmd/vet/testdata", pkg))
	cmd.Env = os.Environ()
	return cmd
}

func TestVet(t *testing.T) {
	t.Parallel()
	Build(t)
	for _, pkg := range []string{
		"asm",
		"assign",
		"atomic",
		"bool",
		"buildtag",
		"cgo",
		"composite",
		"copylock",
		"deadcode",
		"httpresponse",
		"lostcancel",
		"method",
		"nilfunc",
		"print",
		"rangeloop",
		"shift",
		"structtag",
		"testingpkg",
		// "testtag" has its own test
		"unmarshal",
		"unsafeptr",
		"unused",
	} {
		pkg := pkg
		t.Run(pkg, func(t *testing.T) {
			t.Parallel()

			// Skip cgo test on platforms without cgo.
			if pkg == "cgo" && !cgoEnabled(t) {
				return
			}

			cmd := vetCmd(t, "-printfuncs=Warn,Warnf", pkg)

			// The asm test assumes amd64.
			if pkg == "asm" {
				cmd.Env = append(cmd.Env, "GOOS=linux", "GOARCH=amd64")
			}

			dir := filepath.Join("testdata", pkg)
			gos, err := filepath.Glob(filepath.Join(dir, "*.go"))
			if err != nil {
				t.Fatal(err)
			}
			asms, err := filepath.Glob(filepath.Join(dir, "*.s"))
			if err != nil {
				t.Fatal(err)
			}
			var files []string
			files = append(files, gos...)
			files = append(files, asms...)

			errchk(cmd, files, t)
		})
	}
}

func cgoEnabled(t *testing.T) bool {
	// Don't trust build.Default.CgoEnabled as it is false for
	// cross-builds unless CGO_ENABLED is explicitly specified.
	// That's fine for the builders, but causes commands like
	// 'GOARCH=386 go test .' to fail.
	// Instead, we ask the go command.
	cmd := exec.Command(testenv.GoToolPath(t), "list", "-f", "{{context.CgoEnabled}}")
	out, _ := cmd.CombinedOutput()
	return string(out) == "true\n"
}

func errchk(c *exec.Cmd, files []string, t *testing.T) {
	output, err := c.CombinedOutput()
	if _, ok := err.(*exec.ExitError); !ok {
		t.Logf("vet output:\n%s", output)
		t.Fatal(err)
	}
	fullshort := make([]string, 0, len(files)*2)
	for _, f := range files {
		fullshort = append(fullshort, f, filepath.Base(f))
	}
	err = errorCheck(string(output), false, fullshort...)
	if err != nil {
		t.Errorf("error check failed: %s", err)
	}
}

// TestTags verifies that the -tags argument controls which files to check.
func TestTags(t *testing.T) {
	t.Parallel()
	Build(t)
	for tag, wantFile := range map[string]int{
		"testtag":     1, // file1
		"x testtag y": 1,
		"othertag":    2,
	} {
		tag, wantFile := tag, wantFile
		t.Run(tag, func(t *testing.T) {
			t.Parallel()
			t.Logf("-tags=%s", tag)
			cmd := vetCmd(t, "-tags="+tag, "tagtest")
			output, err := cmd.CombinedOutput()

			want := fmt.Sprintf("file%d.go", wantFile)
			dontwant := fmt.Sprintf("file%d.go", 3-wantFile)

			// file1 has testtag and file2 has !testtag.
			if !bytes.Contains(output, []byte(filepath.Join("tagtest", want))) {
				t.Errorf("%s: %s was excluded, should be included", tag, want)
			}
			if bytes.Contains(output, []byte(filepath.Join("tagtest", dontwant))) {
				t.Errorf("%s: %s was included, should be excluded", tag, dontwant)
			}
			if t.Failed() {
				t.Logf("err=%s, output=<<%s>>", err, output)
			}
		})
	}
}

// All declarations below were adapted from test/run.go.

// errorCheck matches errors in outStr against comments in source files.
// For each line of the source files which should generate an error,
// there should be a comment of the form // ERROR "regexp".
// If outStr has an error for a line which has no such comment,
// this function will report an error.
// Likewise if outStr does not have an error for a line which has a comment,
// or if the error message does not match the <regexp>.
// The <regexp> syntax is Perl but it's best to stick to egrep.
//
// Sources files are supplied as fullshort slice.
// It consists of pairs: full path to source file and its base name.
func errorCheck(outStr string, wantAuto bool, fullshort ...string) (err error) {
	var errs []error
	out := splitOutput(outStr, wantAuto)
	// Cut directory name.
	for i := range out {
		for j := 0; j < len(fullshort); j += 2 {
			full, short := fullshort[j], fullshort[j+1]
			out[i] = strings.ReplaceAll(out[i], full, short)
		}
	}

	var want []wantedError
	for j := 0; j < len(fullshort); j += 2 {
		full, short := fullshort[j], fullshort[j+1]
		want = append(want, wantedErrors(full, short)...)
	}
	for _, we := range want {
		var errmsgs []string
		if we.auto {
			errmsgs, out = partitionStrings("<autogenerated>", out)
		} else {
			errmsgs, out = partitionStrings(we.prefix, out)
		}
		if len(errmsgs) == 0 {
			errs = append(errs, fmt.Errorf("%s:%d: missing error %q", we.file, we.lineNum, we.reStr))
			continue
		}
		matched := false
		n := len(out)
		for _, errmsg := range errmsgs {
			// Assume errmsg says "file:line: foo".
			// Cut leading "file:line: " to avoid accidental matching of file name instead of message.
			text := errmsg
			if _, suffix, ok := strings.Cut(text, " "); ok {
				text = suffix
			}
			if we.re.MatchString(text) {
				matched = true
			} else {
				out = append(out, errmsg)
			}
		}
		if !matched {
			errs = append(errs, fmt.Errorf("%s:%d: no match for %#q in:\n\t%s", we.file, we.lineNum, we.reStr, strings.Join(out[n:], "\n\t")))
			continue
		}
	}

	if len(out) > 0 {
		errs = append(errs, fmt.Errorf("Unmatched Errors:"))
		for _, errLine := range out {
			errs = append(errs, fmt.Errorf("%s", errLine))
		}
	}

	if len(errs) == 0 {
		return nil
	}
	if len(errs) == 1 {
		return errs[0]
	}
	var buf strings.Builder
	fmt.Fprintf(&buf, "\n")
	for _, err := range errs {
		fmt.Fprintf(&buf, "%s\n", err.Error())
	}
	return errors.New(buf.String())
}

func splitOutput(out string, wantAuto bool) []string {
	// gc error messages continue onto additional lines with leading tabs.
	// Split the output at the beginning of each line that doesn't begin with a tab.
	// <autogenerated> lines are impossible to match so those are filtered out.
	var res []string
	for _, line := range strings.Split(out, "\n") {
		line = strings.TrimSuffix(line, "\r") // normalize Windows output
		if strings.HasPrefix(line, "\t") {
			res[len(res)-1] += "\n" + line
		} else if strings.HasPrefix(line, "go tool") || strings.HasPrefix(line, "#") || !wantAuto && strings.HasPrefix(line, "<autogenerated>") {
			continue
		} else if strings.TrimSpace(line) != "" {
			res = append(res, line)
		}
	}
	return res
}

// matchPrefix reports whether s starts with file name prefix followed by a :,
// and possibly preceded by a directory name.
func matchPrefix(s, prefix string) bool {
	i := strings.Index(s, ":")
	if i < 0 {
		return false
	}
	j := strings.LastIndex(s[:i], "/")
	s = s[j+1:]
	if len(s) <= len(prefix) || s[:len(prefix)] != prefix {
		return false
	}
	if s[len(prefix)] == ':' {
		return true
	}
	return false
}

func partitionStrings(prefix string, strs []string) (matched, unmatched []string) {
	for _, s := range strs {
		if matchPrefix(s, prefix) {
			matched = append(matched, s)
		} else {
			unmatched = append(unmatched, s)
		}
	}
	return
}

type wantedError struct {
	reStr   string
	re      *regexp.Regexp
	lineNum int
	auto    bool // match <autogenerated> line
	file    string
	prefix  string
}

var (
	errRx       = regexp.MustCompile(`// (?:GC_)?ERROR(NEXT)? (.*)`)
	errAutoRx   = regexp.MustCompile(`// (?:GC_)?ERRORAUTO(NEXT)? (.*)`)
	errQuotesRx = regexp.MustCompile(`"([^"]*)"`)
	lineRx      = regexp.MustCompile(`LINE(([+-])([0-9]+))?`)
)

// wantedErrors parses expected errors from comments in a file.
func wantedErrors(file, short string) (errs []wantedError) {
	cache := make(map[string]*regexp.Regexp)

	src, err := os.ReadFile(file)
	if err != nil {
		log.Fatal(err)
	}
	for i, line := range strings.Split(string(src), "\n") {
		lineNum := i + 1
		if strings.Contains(line, "////") {
			// double comment disables ERROR
			continue
		}
		var auto bool
		m := errAutoRx.FindStringSubmatch(line)
		if m != nil {
			auto = true
		} else {
			m = errRx.FindStringSubmatch(line)
		}
		if m == nil {
			continue
		}
		if m[1] == "NEXT" {
			lineNum++
		}
		all := m[2]
		mm := errQuotesRx.FindAllStringSubmatch(all, -1)
		if mm == nil {
			log.Fatalf("%s:%d: invalid errchk line: %s", file, lineNum, line)
		}
		for _, m := range mm {
			replacedOnce := false
			rx := lineRx.ReplaceAllStringFunc(m[1], func(m string) string {
				if replacedOnce {
					return m
				}
				replacedOnce = true
				n := lineNum
				if strings.HasPrefix(m, "LINE+") {
					delta, _ := strconv.Atoi(m[5:])
					n += delta
				} else if strings.HasPrefix(m, "LINE-") {
					delta, _ := strconv.Atoi(m[5:])
					n -= delta
				}
				return fmt.Sprintf("%s:%d", short, n)
			})
			re := cache[rx]
			if re == nil {
				var err error
				re, err = regexp.Compile(rx)
				if err != nil {
					log.Fatalf("%s:%d: invalid regexp \"%#q\" in ERROR line: %v", file, lineNum, rx, err)
				}
				cache[rx] = re
			}
			prefix := fmt.Sprintf("%s:%d", short, lineNum)
			errs = append(errs, wantedError{
				reStr:   rx,
				re:      re,
				prefix:  prefix,
				auto:    auto,
				lineNum: lineNum,
				file:    short,
			})
		}
	}

	return
}
