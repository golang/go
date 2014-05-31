// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build race

// This program is used to verify the race detector
// by running the tests and parsing their output.
// It does not check stack correctness, completeness or anything else:
// it merely verifies that if a test is expected to be racy
// then the race is detected.
package race_test

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

var (
	passedTests = 0
	totalTests  = 0
	falsePos    = 0
	falseNeg    = 0
	failingPos  = 0
	failingNeg  = 0
	failed      = false
)

const (
	visibleLen = 40
	testPrefix = "=== RUN Test"
)

func TestRace(t *testing.T) {
	testOutput, err := runTests()
	if err != nil {
		t.Fatalf("Failed to run tests: %v\n%v", err, string(testOutput))
	}
	reader := bufio.NewReader(bytes.NewReader(testOutput))

	funcName := ""
	var tsanLog []string
	for {
		s, err := nextLine(reader)
		if err != nil {
			fmt.Printf("%s\n", processLog(funcName, tsanLog))
			break
		}
		if strings.HasPrefix(s, testPrefix) {
			fmt.Printf("%s\n", processLog(funcName, tsanLog))
			tsanLog = make([]string, 0, 100)
			funcName = s[len(testPrefix):]
		} else {
			tsanLog = append(tsanLog, s)
		}
	}

	fmt.Printf("\nPassed %d of %d tests (%.02f%%, %d+, %d-)\n",
		passedTests, totalTests, 100*float64(passedTests)/float64(totalTests), falsePos, falseNeg)
	fmt.Printf("%d expected failures (%d has not fail)\n", failingPos+failingNeg, failingNeg)
	if failed {
		t.Fail()
	}
}

// nextLine is a wrapper around bufio.Reader.ReadString.
// It reads a line up to the next '\n' character. Error
// is non-nil if there are no lines left, and nil
// otherwise.
func nextLine(r *bufio.Reader) (string, error) {
	s, err := r.ReadString('\n')
	if err != nil {
		if err != io.EOF {
			log.Fatalf("nextLine: expected EOF, received %v", err)
		}
		return s, err
	}
	return s[:len(s)-1], nil
}

// processLog verifies whether the given ThreadSanitizer's log
// contains a race report, checks this information against
// the name of the testcase and returns the result of this
// comparison.
func processLog(testName string, tsanLog []string) string {
	if !strings.HasPrefix(testName, "Race") && !strings.HasPrefix(testName, "NoRace") {
		return ""
	}
	gotRace := false
	for _, s := range tsanLog {
		if strings.Contains(s, "DATA RACE") {
			gotRace = true
			break
		}
	}

	failing := strings.Contains(testName, "Failing")
	expRace := !strings.HasPrefix(testName, "No")
	for len(testName) < visibleLen {
		testName += " "
	}
	if expRace == gotRace {
		passedTests++
		totalTests++
		if failing {
			failed = true
			failingNeg++
		}
		return fmt.Sprintf("%s .", testName)
	}
	pos := ""
	if expRace {
		falseNeg++
	} else {
		falsePos++
		pos = "+"
	}
	if failing {
		failingPos++
	} else {
		failed = true
	}
	totalTests++
	return fmt.Sprintf("%s %s%s", testName, "FAILED", pos)
}

// runTests assures that the package and its dependencies is
// built with instrumentation enabled and returns the output of 'go test'
// which includes possible data race reports from ThreadSanitizer.
func runTests() ([]byte, error) {
	tests, err := filepath.Glob("./testdata/*_test.go")
	if err != nil {
		return nil, err
	}
	args := []string{"test", "-race", "-v"}
	args = append(args, tests...)
	cmd := exec.Command("go", args...)
	// The following flags turn off heuristics that suppress seemingly identical reports.
	// It is required because the tests contain a lot of data races on the same addresses
	// (the tests are simple and the memory is constantly reused).
	for _, env := range os.Environ() {
		if strings.HasPrefix(env, "GOMAXPROCS=") || strings.HasPrefix(env, "GODEBUG=") {
			continue
		}
		cmd.Env = append(cmd.Env, env)
	}
	cmd.Env = append(cmd.Env, `GORACE="suppress_equal_stacks=0 suppress_equal_addresses=0 exitcode=0"`)
	return cmd.CombinedOutput()
}

func TestIssue8102(t *testing.T) {
	// If this compiles with -race, the test passes.
	type S struct {
		x interface{}
		i int
	}
	c := make(chan int)
	a := [2]*int{}
	for ; ; c <- *a[S{}.i] {
		if t != nil {
			break
		}
	}
}
