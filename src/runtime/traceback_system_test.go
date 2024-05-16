// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

// This test of GOTRACEBACK=system has its own file,
// to minimize line-number perturbation.

import (
	"bytes"
	"fmt"
	"internal/testenv"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"runtime/debug"
	"strconv"
	"strings"
	"testing"
)

// This is the entrypoint of the child process used by
// TestTracebackSystem. It prints a crash report to stdout.
func crash() {
	// Ensure that we get pc=0x%x values in the traceback.
	debug.SetTraceback("system")
	writeSentinel(os.Stdout)
	debug.SetCrashOutput(os.Stdout, debug.CrashOptions{})

	go func() {
		// This call is typically inlined.
		child1()
	}()
	select {}
}

func child1() {
	child2()
}

func child2() {
	child3()
}

func child3() {
	child4()
}

func child4() {
	child5()
}

//go:noinline
func child5() { // test trace through second of two call instructions
	child6bad()
	child6() // appears in stack trace
}

//go:noinline
func child6bad() {
}

//go:noinline
func child6() { // test trace through first of two call instructions
	child7() // appears in stack trace
	child7bad()
}

//go:noinline
func child7bad() {
}

//go:noinline
func child7() {
	// Write runtime.Caller's view of the stack to stderr, for debugging.
	var pcs [16]uintptr
	n := runtime.Callers(1, pcs[:])
	fmt.Fprintf(os.Stderr, "Callers: %#x\n", pcs[:n])
	io.WriteString(os.Stderr, formatStack(pcs[:n]))

	// Cause the crash report to be written to stdout.
	panic("oops")
}

// TestTracebackSystem tests that the syntax of crash reports produced
// by GOTRACEBACK=system (see traceback2) contains a complete,
// parseable list of program counters for the running goroutine that
// can be parsed and fed to runtime.CallersFrames to obtain accurate
// information about the logical call stack, even in the presence of
// inlining.
//
// The test is a distillation of the crash monitor in
// golang.org/x/telemetry/crashmonitor.
func TestTracebackSystem(t *testing.T) {
	testenv.MustHaveExec(t)
	if runtime.GOOS == "android" {
		t.Skip("Can't read source code for this file on Android")
	}

	// Fork+exec the crashing process.
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	cmd := testenv.Command(t, exe)
	cmd.Env = append(cmd.Environ(), entrypointVar+"=crash")
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	cmd.Run() // expected to crash
	t.Logf("stderr:\n%s\nstdout: %s\n", stderr.Bytes(), stdout.Bytes())
	crash := stdout.String()

	// If the only line is the sentinel, it wasn't a crash.
	if strings.Count(crash, "\n") < 2 {
		t.Fatalf("child process did not produce a crash report")
	}

	// Parse the PCs out of the child's crash report.
	pcs, err := parseStackPCs(crash)
	if err != nil {
		t.Fatal(err)
	}

	// Unwind the stack using this executable's symbol table.
	got := formatStack(pcs)
	want := `redacted.go:0: runtime.gopanic
traceback_system_test.go:85: runtime_test.child7: 	panic("oops")
traceback_system_test.go:68: runtime_test.child6: 	child7() // appears in stack trace
traceback_system_test.go:59: runtime_test.child5: 	child6() // appears in stack trace
traceback_system_test.go:53: runtime_test.child4: 	child5()
traceback_system_test.go:49: runtime_test.child3: 	child4()
traceback_system_test.go:45: runtime_test.child2: 	child3()
traceback_system_test.go:41: runtime_test.child1: 	child2()
traceback_system_test.go:35: runtime_test.crash.func1: 		child1()
redacted.go:0: runtime.goexit
`
	if strings.TrimSpace(got) != strings.TrimSpace(want) {
		t.Errorf("got:\n%swant:\n%s", got, want)
	}
}

// parseStackPCs parses the parent process's program counters for the
// first running goroutine out of a GOTRACEBACK=system traceback,
// adjusting them so that they are valid for the child process's text
// segment.
//
// This function returns only program counter values, ensuring that
// there is no possibility of strings from the crash report (which may
// contain PII) leaking into the telemetry system.
//
// (Copied from golang.org/x/telemetry/crashmonitor.parseStackPCs.)
func parseStackPCs(crash string) ([]uintptr, error) {
	// getPC parses the PC out of a line of the form:
	//     \tFILE:LINE +0xRELPC sp=... fp=... pc=...
	getPC := func(line string) (uint64, error) {
		_, pcstr, ok := strings.Cut(line, " pc=") // e.g. pc=0x%x
		if !ok {
			return 0, fmt.Errorf("no pc= for stack frame: %s", line)
		}
		return strconv.ParseUint(pcstr, 0, 64) // 0 => allow 0x prefix
	}

	var (
		pcs            []uintptr
		parentSentinel uint64
		childSentinel  = sentinel()
		on             = false // are we in the first running goroutine?
		lines          = strings.Split(crash, "\n")
	)
	for i := 0; i < len(lines); i++ {
		line := lines[i]

		// Read sentinel value.
		if parentSentinel == 0 && strings.HasPrefix(line, "sentinel ") {
			_, err := fmt.Sscanf(line, "sentinel %x", &parentSentinel)
			if err != nil {
				return nil, fmt.Errorf("can't read sentinel line")
			}
			continue
		}

		// Search for "goroutine GID [STATUS]"
		if !on {
			if strings.HasPrefix(line, "goroutine ") &&
				strings.Contains(line, " [running]:") {
				on = true

				if parentSentinel == 0 {
					return nil, fmt.Errorf("no sentinel value in crash report")
				}
			}
			continue
		}

		// A blank line marks end of a goroutine stack.
		if line == "" {
			break
		}

		// Skip the final "created by SYMBOL in goroutine GID" part.
		if strings.HasPrefix(line, "created by ") {
			break
		}

		// Expect a pair of lines:
		//   SYMBOL(ARGS)
		//   \tFILE:LINE +0xRELPC sp=0x%x fp=0x%x pc=0x%x
		// Note: SYMBOL may contain parens "pkg.(*T).method"
		// The RELPC is sometimes missing.

		// Skip the symbol(args) line.
		i++
		if i == len(lines) {
			break
		}
		line = lines[i]

		// Parse the PC, and correct for the parent and child's
		// different mappings of the text section.
		pc, err := getPC(line)
		if err != nil {
			// Inlined frame, perhaps; skip it.
			continue
		}
		pcs = append(pcs, uintptr(pc-parentSentinel+childSentinel))
	}
	return pcs, nil
}

// The sentinel function returns its address. The difference between
// this value as observed by calls in two different processes of the
// same executable tells us the relative offset of their text segments.
//
// It would be nice if SetCrashOutput took care of this as it's fiddly
// and likely to confuse every user at first.
func sentinel() uint64 {
	return uint64(reflect.ValueOf(sentinel).Pointer())
}

func writeSentinel(out io.Writer) {
	fmt.Fprintf(out, "sentinel %x\n", sentinel())
}

// formatStack formats a stack of PC values using the symbol table,
// redacting information that cannot be relied upon in the test.
func formatStack(pcs []uintptr) string {
	// When debugging, show file/line/content of files other than this one.
	const debug = false

	var buf strings.Builder
	i := 0
	frames := runtime.CallersFrames(pcs)
	for {
		fr, more := frames.Next()
		if debug {
			fmt.Fprintf(&buf, "pc=%x ", pcs[i])
			i++
		}
		if base := filepath.Base(fr.File); base == "traceback_system_test.go" || debug {
			content, err := os.ReadFile(fr.File)
			if err != nil {
				panic(err)
			}
			lines := bytes.Split(content, []byte("\n"))
			fmt.Fprintf(&buf, "%s:%d: %s: %s\n", base, fr.Line, fr.Function, lines[fr.Line-1])
		} else {
			// For robustness, don't show file/line for functions from other files.
			fmt.Fprintf(&buf, "redacted.go:0: %s\n", fr.Function)
		}

		if !more {
			break
		}
	}
	return buf.String()
}
