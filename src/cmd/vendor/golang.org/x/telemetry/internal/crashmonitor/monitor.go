// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package crashmonitor

// This file defines a monitor that reports arbitrary Go runtime
// crashes to telemetry.

import (
	"bytes"
	"fmt"
	"io"
	"log"
	"os"
	"reflect"
	"runtime/debug"
	"strconv"
	"strings"

	"golang.org/x/telemetry/internal/counter"
)

// Supported reports whether the runtime supports [runtime.SetCrashOutput].
//
// TODO(adonovan): eliminate once go1.23+ is assured.
func Supported() bool { return setCrashOutput != nil }

var setCrashOutput func(*os.File) error // = runtime.SetCrashOutput on go1.23+

// Parent sets up the parent side of the crashmonitor. It requires
// exclusive use of a writable pipe connected to the child process's stdin.
func Parent(pipe *os.File) {
	writeSentinel(pipe)
	// Ensure that we get pc=0x%x values in the traceback.
	debug.SetTraceback("system")
	setCrashOutput(pipe)
}

// Child runs the part of the crashmonitor that runs in the child process.
// It expects its stdin to be connected via a pipe to the parent which has
// run Parent.
func Child() {
	// Wait for parent process's dying gasp.
	// If the parent dies for any reason this read will return.
	data, err := io.ReadAll(os.Stdin)
	if err != nil {
		log.Fatalf("failed to read from input pipe: %v", err)
	}

	// If the only line is the sentinel, it wasn't a crash.
	if bytes.Count(data, []byte("\n")) < 2 {
		childExitHook()
		os.Exit(0) // parent exited without crash report
	}

	log.Printf("parent reported crash:\n%s", data)

	// Parse the stack out of the crash report
	// and record a telemetry count for it.
	name, err := telemetryCounterName(data)
	if err != nil {
		// Keep count of how often this happens
		// so that we can investigate if necessary.
		incrementCounter("crash/malformed")

		// Something went wrong.
		// Save the crash securely in the file system.
		f, err := os.CreateTemp(os.TempDir(), "*.crash")
		if err != nil {
			log.Fatal(err)
		}
		if _, err := f.Write(data); err != nil {
			log.Fatal(err)
		}
		if err := f.Close(); err != nil {
			log.Fatal(err)
		}
		log.Printf("failed to report crash to telemetry: %v", err)
		log.Fatalf("crash report saved at %s", f.Name())
	}

	incrementCounter(name)

	childExitHook()
	log.Fatalf("telemetry crash recorded")
}

// (stubbed by test)
var (
	incrementCounter = func(name string) { counter.New(name).Inc() }
	childExitHook    = func() {}
)

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

// telemetryCounterName parses a crash report produced by the Go
// runtime, extracts the stack of the first runnable goroutine,
// converts each line into telemetry form ("symbol:relative-line"),
// and returns this as the name of a counter.
func telemetryCounterName(crash []byte) (string, error) {
	pcs, err := parseStackPCs(string(crash))
	if err != nil {
		return "", err
	}

	// Limit the number of frames we request.
	pcs = pcs[:min(len(pcs), 16)]

	if len(pcs) == 0 {
		// This can occur if all goroutines are idle, as when
		// caught in a deadlock, or killed by an async signal
		// while blocked.
		//
		// TODO(adonovan): consider how to report such
		// situations. Reporting a goroutine in [sleep] or
		// [select] state could be quite confusing without
		// further information about the nature of the crash,
		// as the problem is not local to the code location.
		//
		// For now, we keep count of this situation so that we
		// can access whether it needs a more involved solution.
		return "crash/no-running-goroutine", nil
	}

	// This string appears at the start of all
	// crashmonitor-generated counter names.
	//
	// It is tempting to expose this as a parameter of Start, but
	// it is not without risk. What value should most programs
	// provide? There's no point giving the name of the executable
	// as this is already recorded by telemetry. What if the
	// application runs in multiple modes? Then it might be useful
	// to record the mode. The problem is that an application with
	// multiple modes probably doesn't know its mode by line 1 of
	// main.main: it might require flag or argument parsing, or
	// even validation of an environment variable, and we really
	// want to steer users aware from any logic before Start. The
	// flags and arguments will be wrong in the child process, and
	// every extra conditional branch creates a risk that the
	// recursively executed child program will behave not like the
	// monitor but like the application. If the child process
	// exits before calling Start, then the parent application
	// will not have a monitor, and its crash reports will be
	// discarded (written in to a pipe that is never read).
	//
	// So for now, we use this constant string.
	const prefix = "crash/crash"
	return counter.EncodeStack(pcs, prefix), nil
}

// parseStackPCs parses the parent process's program counters for the
// first running goroutine out of a GOTRACEBACK=system traceback,
// adjusting them so that they are valid for the child process's text
// segment.
//
// This function returns only program counter values, ensuring that
// there is no possibility of strings from the crash report (which may
// contain PII) leaking into the telemetry system.
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

func min(x, y int) int {
	if x < y {
		return x
	} else {
		return y
	}
}
