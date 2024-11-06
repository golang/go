// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build race

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
	"internal/testenv"
	"io"
	"log"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
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
	testPrefix = "=== RUN   Test"
)

func TestRace(t *testing.T) {
	testOutput, err := runTests(t)
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

	if totalTests == 0 {
		t.Fatalf("failed to parse test output:\n%s", testOutput)
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
		if strings.Contains(s, "fatal error: concurrent map") {
			// Detected by the runtime, not the race detector.
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
func runTests(t *testing.T) ([]byte, error) {
	tests, err := filepath.Glob("./testdata/*_test.go")
	if err != nil {
		return nil, err
	}
	args := []string{"test", "-race", "-v"}
	args = append(args, tests...)
	cmd := exec.Command(testenv.GoToolPath(t), args...)
	// The following flags turn off heuristics that suppress seemingly identical reports.
	// It is required because the tests contain a lot of data races on the same addresses
	// (the tests are simple and the memory is constantly reused).
	for _, env := range os.Environ() {
		if strings.HasPrefix(env, "GOMAXPROCS=") ||
			strings.HasPrefix(env, "GODEBUG=") ||
			strings.HasPrefix(env, "GORACE=") {
			continue
		}
		cmd.Env = append(cmd.Env, env)
	}
	// We set GOMAXPROCS=1 to prevent test flakiness.
	// There are two sources of flakiness:
	// 1. Some tests rely on particular execution order.
	//    If the order is different, race does not happen at all.
	// 2. Ironically, ThreadSanitizer runtime contains a logical race condition
	//    that can lead to false negatives if racy accesses happen literally at the same time.
	// Tests used to work reliably in the good old days of GOMAXPROCS=1.
	// So let's set it for now. A more reliable solution is to explicitly annotate tests
	// with required execution order by means of a special "invisible" synchronization primitive
	// (that's what is done for C++ ThreadSanitizer tests). This is issue #14119.
	cmd.Env = append(cmd.Env,
		"GOMAXPROCS=1",
		"GORACE=suppress_equal_stacks=0 suppress_equal_addresses=0",
	)
	// There are races: we expect tests to fail and the exit code to be non-zero.
	out, _ := cmd.CombinedOutput()
	fatals := bytes.Count(out, []byte("fatal error:"))
	mapFatals := bytes.Count(out, []byte("fatal error: concurrent map"))
	if fatals > mapFatals {
		// But don't expect runtime to crash (other than
		// in the map concurrent access detector).
		return out, fmt.Errorf("runtime fatal error")
	}
	return out, nil
}

func TestIssue8102(t *testing.T) {
	// If this compiles with -race, the test passes.
	type S struct {
		x any
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

func TestIssue9137(t *testing.T) {
	a := []string{"a"}
	i := 0
	a[i], a[len(a)-1], a = a[len(a)-1], "", a[:len(a)-1]
	if len(a) != 0 || a[:1][0] != "" {
		t.Errorf("mangled a: %q %q", a, a[:1])
	}
}

func BenchmarkSyncLeak(b *testing.B) {
	const (
		G = 1000
		S = 1000
		H = 10
	)
	var wg sync.WaitGroup
	wg.Add(G)
	for g := 0; g < G; g++ {
		go func() {
			defer wg.Done()
			hold := make([][]uint32, H)
			for i := 0; i < b.N; i++ {
				a := make([]uint32, S)
				atomic.AddUint32(&a[rand.Intn(len(a))], 1)
				hold[rand.Intn(len(hold))] = a
			}
			_ = hold
		}()
	}
	wg.Wait()
}

func BenchmarkStackLeak(b *testing.B) {
	done := make(chan bool, 1)
	for i := 0; i < b.N; i++ {
		go func() {
			growStack(rand.Intn(100))
			done <- true
		}()
		<-done
	}
}

func growStack(i int) {
	if i == 0 {
		return
	}
	growStack(i - 1)
}
