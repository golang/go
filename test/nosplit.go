// run

//go:build !nacl && !js && !aix && !openbsd && !wasip1 && !gcflags_noopt && gc

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
)

const debug = false

var tests = `
# These are test cases for the linker analysis that detects chains of
# nosplit functions that would cause a stack overflow.
#
# Lines beginning with # are comments.
#
# Each test case describes a sequence of functions, one per line.
# Each function definition is the function name, then the frame size,
# then optionally the keyword 'nosplit', then the body of the function.
# The body is assembly code, with some shorthands.
# The shorthand 'call x' stands for CALL x(SB).
# The shorthand 'callind' stands for 'CALL R0', where R0 is a register.
# Each test case must define a function named start, and it must be first.
# That is, a line beginning "start " indicates the start of a new test case.
# Within a stanza, ; can be used instead of \n to separate lines.
#
# After the function definition, the test case ends with an optional
# REJECT line, specifying the architectures on which the case should
# be rejected. "REJECT" without any architectures means reject on all architectures.
# The linker should accept the test case on systems not explicitly rejected.
#
# 64-bit systems do not attempt to execute test cases with frame sizes
# that are only 32-bit aligned.

# Ordinary function should work
start 0

# Large frame marked nosplit is always wrong.
# Frame is so large it overflows cmd/link's int16.
start 100000 nosplit
REJECT

# Calling a large frame is okay.
start 0 call big
big 10000

# But not if the frame is nosplit.
start 0 call big
big 10000 nosplit
REJECT

# Recursion is okay.
start 0 call start

# Recursive nosplit runs out of space.
start 0 nosplit call start
REJECT

# Non-trivial recursion runs out of space.
start 0 call f1
f1 0 nosplit call f2
f2 0 nosplit call f1
REJECT
# Same but cycle starts below nosplit entry.
start 0 call f1
f1 0 nosplit call f2
f2 0 nosplit call f3
f3 0 nosplit call f2
REJECT

# Chains of ordinary functions okay.
start 0 call f1
f1 80 call f2
f2 80

# Chains of nosplit must fit in the stack limit, 128 bytes.
start 0 call f1
f1 80 nosplit call f2
f2 80 nosplit
REJECT

# Larger chains.
start 0 call f1
f1 16 call f2
f2 16 call f3
f3 16 call f4
f4 16 call f5
f5 16 call f6
f6 16 call f7
f7 16 call f8
f8 16 call end
end 1000

start 0 call f1
f1 16 nosplit call f2
f2 16 nosplit call f3
f3 16 nosplit call f4
f4 16 nosplit call f5
f5 16 nosplit call f6
f6 16 nosplit call f7
f7 16 nosplit call f8
f8 16 nosplit call end
end 1000
REJECT

# Two paths both go over the stack limit.
start 0 call f1
f1 80 nosplit call f2 call f3
f2 40 nosplit call f4
f3 96 nosplit
f4 40 nosplit
REJECT

# Test cases near the 128-byte limit.

# Ordinary stack split frame is always okay.
start 112
start 116
start 120
start 124
start 128
start 132
start 136

# A nosplit leaf can use the whole 128-CallSize bytes available on entry.
# (CallSize is 32 on ppc64, 8 on amd64 for frame pointer.)
start 96 nosplit
start 100 nosplit; REJECT ppc64 ppc64le
start 104 nosplit; REJECT ppc64 ppc64le arm64
start 108 nosplit; REJECT ppc64 ppc64le
start 112 nosplit; REJECT ppc64 ppc64le arm64
start 116 nosplit; REJECT ppc64 ppc64le
start 120 nosplit; REJECT ppc64 ppc64le amd64 arm64
start 124 nosplit; REJECT ppc64 ppc64le amd64
start 128 nosplit; REJECT
start 132 nosplit; REJECT
start 136 nosplit; REJECT

# Calling a nosplit function from a nosplit function requires
# having room for the saved caller PC and the called frame.
# Because ARM doesn't save LR in the leaf, it gets an extra 4 bytes.
# Because arm64 doesn't save LR in the leaf, it gets an extra 8 bytes.
# ppc64 doesn't save LR in the leaf, but CallSize is 32, so it gets 24 bytes.
# Because AMD64 uses frame pointer, it has 8 fewer bytes.
start 96 nosplit call f; f 0 nosplit
start 100 nosplit call f; f 0 nosplit; REJECT ppc64 ppc64le
start 104 nosplit call f; f 0 nosplit; REJECT ppc64 ppc64le arm64
start 108 nosplit call f; f 0 nosplit; REJECT ppc64 ppc64le
start 112 nosplit call f; f 0 nosplit; REJECT ppc64 ppc64le amd64 arm64
start 116 nosplit call f; f 0 nosplit; REJECT ppc64 ppc64le amd64
start 120 nosplit call f; f 0 nosplit; REJECT ppc64 ppc64le amd64 arm64
start 124 nosplit call f; f 0 nosplit; REJECT ppc64 ppc64le amd64 386
start 128 nosplit call f; f 0 nosplit; REJECT
start 132 nosplit call f; f 0 nosplit; REJECT
start 136 nosplit call f; f 0 nosplit; REJECT

# Calling a splitting function from a nosplit function requires
# having room for the saved caller PC of the call but also the
# saved caller PC for the call to morestack.
# Architectures differ in the same way as before.
start 96 nosplit call f; f 0 call f
start 100 nosplit call f; f 0 call f; REJECT ppc64 ppc64le
start 104 nosplit call f; f 0 call f; REJECT ppc64 ppc64le amd64 arm64
start 108 nosplit call f; f 0 call f; REJECT ppc64 ppc64le amd64
start 112 nosplit call f; f 0 call f; REJECT ppc64 ppc64le amd64 arm64
start 116 nosplit call f; f 0 call f; REJECT ppc64 ppc64le amd64
start 120 nosplit call f; f 0 call f; REJECT ppc64 ppc64le amd64 386 arm64
start 124 nosplit call f; f 0 call f; REJECT ppc64 ppc64le amd64 386
start 128 nosplit call f; f 0 call f; REJECT
start 132 nosplit call f; f 0 call f; REJECT
start 136 nosplit call f; f 0 call f; REJECT

# Indirect calls are assumed to be splitting functions.
start 96 nosplit callind
start 100 nosplit callind; REJECT ppc64 ppc64le
start 104 nosplit callind; REJECT ppc64 ppc64le amd64 arm64
start 108 nosplit callind; REJECT ppc64 ppc64le amd64
start 112 nosplit callind; REJECT ppc64 ppc64le amd64 arm64
start 116 nosplit callind; REJECT ppc64 ppc64le amd64
start 120 nosplit callind; REJECT ppc64 ppc64le amd64 386 arm64
start 124 nosplit callind; REJECT ppc64 ppc64le amd64 386
start 128 nosplit callind; REJECT
start 132 nosplit callind; REJECT
start 136 nosplit callind; REJECT

# Issue 7623
start 0 call f; f 112
start 0 call f; f 116
start 0 call f; f 120
start 0 call f; f 124
start 0 call f; f 128
start 0 call f; f 132
start 0 call f; f 136
`

var (
	commentRE = regexp.MustCompile(`(?m)^#.*`)
	rejectRE  = regexp.MustCompile(`(?s)\A(.+?)((\n|; *)REJECT(.*))?\z`)
	lineRE    = regexp.MustCompile(`(\w+) (\d+)( nosplit)?(.*)`)
	callRE    = regexp.MustCompile(`\bcall (\w+)\b`)
	callindRE = regexp.MustCompile(`\bcallind\b`)
)

func main() {
	goarch := os.Getenv("GOARCH")
	if goarch == "" {
		goarch = runtime.GOARCH
	}

	dir, err := ioutil.TempDir("", "go-test-nosplit")
	if err != nil {
		bug()
		fmt.Printf("creating temp dir: %v\n", err)
		return
	}
	defer os.RemoveAll(dir)
	os.Setenv("GOPATH", filepath.Join(dir, "_gopath"))

	if err := ioutil.WriteFile(filepath.Join(dir, "go.mod"), []byte("module go-test-nosplit\n"), 0666); err != nil {
		log.Panic(err)
	}

	tests = strings.Replace(tests, "\t", " ", -1)
	tests = commentRE.ReplaceAllString(tests, "")

	nok := 0
	nfail := 0
TestCases:
	for len(tests) > 0 {
		var stanza string
		i := strings.Index(tests, "\nstart ")
		if i < 0 {
			stanza, tests = tests, ""
		} else {
			stanza, tests = tests[:i], tests[i+1:]
		}

		m := rejectRE.FindStringSubmatch(stanza)
		if m == nil {
			bug()
			fmt.Printf("invalid stanza:\n\t%s\n", indent(stanza))
			continue
		}
		lines := strings.TrimSpace(m[1])
		reject := false
		if m[2] != "" {
			if strings.TrimSpace(m[4]) == "" {
				reject = true
			} else {
				for _, rej := range strings.Fields(m[4]) {
					if rej == goarch {
						reject = true
					}
				}
			}
		}
		if lines == "" && !reject {
			continue
		}

		var gobuf bytes.Buffer
		fmt.Fprintf(&gobuf, "package main\n")

		var buf bytes.Buffer
		ptrSize := 4
		switch goarch {
		case "mips", "mipsle":
			fmt.Fprintf(&buf, "#define REGISTER (R0)\n")
		case "mips64", "mips64le":
			ptrSize = 8
			fmt.Fprintf(&buf, "#define REGISTER (R0)\n")
		case "loong64":
			ptrSize = 8
			fmt.Fprintf(&buf, "#define REGISTER (R0)\n")
		case "ppc64", "ppc64le":
			ptrSize = 8
			fmt.Fprintf(&buf, "#define REGISTER (CTR)\n")
		case "arm":
			fmt.Fprintf(&buf, "#define REGISTER (R0)\n")
		case "arm64":
			ptrSize = 8
			fmt.Fprintf(&buf, "#define REGISTER (R0)\n")
		case "amd64":
			ptrSize = 8
			fmt.Fprintf(&buf, "#define REGISTER AX\n")
		case "riscv64":
			ptrSize = 8
			fmt.Fprintf(&buf, "#define REGISTER A0\n")
		case "s390x":
			ptrSize = 8
			fmt.Fprintf(&buf, "#define REGISTER R10\n")
		default:
			fmt.Fprintf(&buf, "#define REGISTER AX\n")
		}

		// Since all of the functions we're generating are
		// ABI0, first enter ABI0 via a splittable function
		// and then go to the chain we're testing. This way we
		// don't have to account for ABI wrappers in the chain.
		fmt.Fprintf(&gobuf, "func main0()\n")
		fmt.Fprintf(&gobuf, "func main() { main0() }\n")
		fmt.Fprintf(&buf, "TEXT ·main0(SB),0,$0-0\n\tCALL ·start(SB)\n")

		adjusted := false
		for _, line := range strings.Split(lines, "\n") {
			line = strings.TrimSpace(line)
			if line == "" {
				continue
			}
			for _, subline := range strings.Split(line, ";") {
				subline = strings.TrimSpace(subline)
				if subline == "" {
					continue
				}
				m := lineRE.FindStringSubmatch(subline)
				if m == nil {
					bug()
					fmt.Printf("invalid function line: %s\n", subline)
					continue TestCases
				}
				name := m[1]
				size, _ := strconv.Atoi(m[2])

				if size%ptrSize == 4 {
					continue TestCases
				}
				nosplit := m[3]
				body := m[4]

				// The limit was originally 128 but is now 800.
				// Instead of rewriting the test cases above, adjust
				// the first nosplit frame to use up the extra bytes.
				// This isn't exactly right because we could have
				// nosplit -> split -> nosplit, but it's good enough.
				if !adjusted && nosplit != "" {
					const stackNosplitBase = 800 // internal/abi.StackNosplitBase
					adjusted = true
					size += stackNosplitBase - 128
				}

				if nosplit != "" {
					nosplit = ",7"
				} else {
					nosplit = ",0"
				}
				body = callRE.ReplaceAllString(body, "CALL ·$1(SB);")
				body = callindRE.ReplaceAllString(body, "CALL REGISTER;")

				fmt.Fprintf(&gobuf, "func %s()\n", name)
				fmt.Fprintf(&buf, "TEXT ·%s(SB)%s,$%d-0\n\t%s\n\tRET\n\n", name, nosplit, size, body)
			}
		}

		if debug {
			fmt.Printf("===\n%s\n", strings.TrimSpace(stanza))
			fmt.Printf("-- main.go --\n%s", gobuf.String())
			fmt.Printf("-- asm.s --\n%s", buf.String())
		}

		if err := ioutil.WriteFile(filepath.Join(dir, "asm.s"), buf.Bytes(), 0666); err != nil {
			log.Fatal(err)
		}
		if err := ioutil.WriteFile(filepath.Join(dir, "main.go"), gobuf.Bytes(), 0666); err != nil {
			log.Fatal(err)
		}

		cmd := exec.Command("go", "build")
		cmd.Dir = dir
		output, err := cmd.CombinedOutput()
		if err == nil {
			nok++
			if reject {
				bug()
				fmt.Printf("accepted incorrectly:\n\t%s\n", indent(strings.TrimSpace(stanza)))
			}
		} else {
			nfail++
			if !reject {
				bug()
				fmt.Printf("rejected incorrectly:\n\t%s\n", indent(strings.TrimSpace(stanza)))
				fmt.Printf("\n\tlinker output:\n\t%s\n", indent(string(output)))
			}
		}
	}

	if !bugged && (nok == 0 || nfail == 0) {
		bug()
		fmt.Printf("not enough test cases run\n")
	}
}

func indent(s string) string {
	return strings.Replace(s, "\n", "\n\t", -1)
}

var bugged = false

func bug() {
	if !bugged {
		bugged = true
		fmt.Printf("BUG\n")
	}
}
