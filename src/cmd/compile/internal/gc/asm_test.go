// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"bytes"
	"fmt"
	"internal/testenv"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"testing"
)

// TestAssembly checks to make sure the assembly generated for
// functions contains certain expected instructions.
func TestAssembly(t *testing.T) {
	if testing.Short() {
		t.Skip("slow test; skipping")
	}
	testenv.MustHaveGoBuild(t)
	if runtime.GOOS == "windows" {
		// TODO: remove if we can get "go tool compile -S" to work on windows.
		t.Skipf("skipping test: recursive windows compile not working")
	}
	dir, err := ioutil.TempDir("", "TestAssembly")
	if err != nil {
		t.Fatalf("could not create directory: %v", err)
	}
	defer os.RemoveAll(dir)

	for _, test := range asmTests {
		asm := compileToAsm(t, dir, test.arch, test.os, fmt.Sprintf(template, test.function))
		// Get rid of code for "".init. Also gets rid of type algorithms & other junk.
		if i := strings.Index(asm, "\n\"\".init "); i >= 0 {
			asm = asm[:i+1]
		}
		for _, r := range test.regexps {
			if b, err := regexp.MatchString(r, asm); !b || err != nil {
				t.Errorf("expected:%s\ngo:%s\nasm:%s\n", r, test.function, asm)
			}
		}
	}
}

// compile compiles the package pkg for architecture arch and
// returns the generated assembly.  dir is a scratch directory.
func compileToAsm(t *testing.T, dir, goarch, goos, pkg string) string {
	// Create source.
	src := filepath.Join(dir, "test.go")
	f, err := os.Create(src)
	if err != nil {
		panic(err)
	}
	f.Write([]byte(pkg))
	f.Close()

	// First, install any dependencies we need.  This builds the required export data
	// for any packages that are imported.
	// TODO: extract dependencies automatically?
	var stdout, stderr bytes.Buffer
	cmd := exec.Command(testenv.GoToolPath(t), "build", "-o", filepath.Join(dir, "encoding/binary.a"), "encoding/binary")
	cmd.Env = mergeEnvLists([]string{"GOARCH=" + goarch, "GOOS=" + goos}, os.Environ())
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		panic(err)
	}
	if s := stdout.String(); s != "" {
		panic(fmt.Errorf("Stdout = %s\nWant empty", s))
	}
	if s := stderr.String(); s != "" {
		panic(fmt.Errorf("Stderr = %s\nWant empty", s))
	}

	// Now, compile the individual file for which we want to see the generated assembly.
	cmd = exec.Command(testenv.GoToolPath(t), "tool", "compile", "-I", dir, "-S", "-o", filepath.Join(dir, "out.o"), src)
	cmd.Env = mergeEnvLists([]string{"GOARCH=" + goarch, "GOOS=" + goos}, os.Environ())
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		panic(err)
	}
	if s := stderr.String(); s != "" {
		panic(fmt.Errorf("Stderr = %s\nWant empty", s))
	}
	return stdout.String()
}

// template to convert a function to a full file
const template = `
package main
%s
`

type asmTest struct {
	// architecture to compile to
	arch string
	// os to compile to
	os string
	// function to compile
	function string
	// regexps that must match the generated assembly
	regexps []string
}

var asmTests = [...]asmTest{
	{"amd64", "linux", `
func f(x int) int {
	return x * 64
}
`,
		[]string{"\tSHLQ\t\\$6,"},
	},
	{"amd64", "linux", `
func f(x int) int {
	return x * 96
}`,
		[]string{"\tSHLQ\t\\$5,", "\tLEAQ\t\\(.*\\)\\(.*\\*2\\),"},
	},
	// Load-combining tests.
	{"amd64", "linux", `
import "encoding/binary"
func f(b []byte) uint64 {
	return binary.LittleEndian.Uint64(b)
}
`,
		[]string{"\tMOVQ\t\\(.*\\),"},
	},
	{"amd64", "linux", `
import "encoding/binary"
func f(b []byte, i int) uint64 {
	return binary.LittleEndian.Uint64(b[i:])
}
`,
		[]string{"\tMOVQ\t\\(.*\\)\\(.*\\*1\\),"},
	},
	{"amd64", "linux", `
import "encoding/binary"
func f(b []byte) uint32 {
	return binary.LittleEndian.Uint32(b)
}
`,
		[]string{"\tMOVL\t\\(.*\\),"},
	},
	{"amd64", "linux", `
import "encoding/binary"
func f(b []byte, i int) uint32 {
	return binary.LittleEndian.Uint32(b[i:])
}
`,
		[]string{"\tMOVL\t\\(.*\\)\\(.*\\*1\\),"},
	},
	{"amd64", "linux", `
import "encoding/binary"
func f(b []byte) uint64 {
	return binary.BigEndian.Uint64(b)
}
`,
		[]string{"\tBSWAPQ\t"},
	},
	{"amd64", "linux", `
import "encoding/binary"
func f(b []byte, i int) uint64 {
	return binary.BigEndian.Uint64(b[i:])
}
`,
		[]string{"\tBSWAPQ\t"},
	},
	{"amd64", "linux", `
import "encoding/binary"
func f(b []byte, v uint64) {
	binary.BigEndian.PutUint64(b, v)
}
`,
		[]string{"\tBSWAPQ\t"},
	},
	{"amd64", "linux", `
import "encoding/binary"
func f(b []byte) uint32 {
	return binary.BigEndian.Uint32(b)
}
`,
		[]string{"\tBSWAPL\t"},
	},
	{"amd64", "linux", `
import "encoding/binary"
func f(b []byte, i int) uint32 {
	return binary.BigEndian.Uint32(b[i:])
}
`,
		[]string{"\tBSWAPL\t"},
	},
	{"amd64", "linux", `
import "encoding/binary"
func f(b []byte, v uint32) {
	binary.BigEndian.PutUint32(b, v)
}
`,
		[]string{"\tBSWAPL\t"},
	},
	{"386", "linux", `
import "encoding/binary"
func f(b []byte) uint32 {
	return binary.LittleEndian.Uint32(b)
}
`,
		[]string{"\tMOVL\t\\(.*\\),"},
	},
	{"386", "linux", `
import "encoding/binary"
func f(b []byte, i int) uint32 {
	return binary.LittleEndian.Uint32(b[i:])
}
`,
		[]string{"\tMOVL\t\\(.*\\)\\(.*\\*1\\),"},
	},

	// Structure zeroing.  See issue #18370.
	{"amd64", "linux", `
type T struct {
	a, b, c int
}
func f(t *T) {
	*t = T{}
}
`,
		[]string{"\tMOVQ\t\\$0, \\(.*\\)", "\tMOVQ\t\\$0, 8\\(.*\\)", "\tMOVQ\t\\$0, 16\\(.*\\)"},
	},
	// TODO: add a test for *t = T{3,4,5} when we fix that.
}

// mergeEnvLists merges the two environment lists such that
// variables with the same name in "in" replace those in "out".
// This always returns a newly allocated slice.
func mergeEnvLists(in, out []string) []string {
	out = append([]string(nil), out...)
NextVar:
	for _, inkv := range in {
		k := strings.SplitAfterN(inkv, "=", 2)[0]
		for i, outkv := range out {
			if strings.HasPrefix(outkv, k) {
				out[i] = inkv
				continue NextVar
			}
		}
		out = append(out, inkv)
	}
	return out
}

// TestLineNumber checks to make sure the generated assembly has line numbers
// see issue #16214
func TestLineNumber(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	dir, err := ioutil.TempDir("", "TestLineNumber")
	if err != nil {
		t.Fatalf("could not create directory: %v", err)
	}
	defer os.RemoveAll(dir)

	src := filepath.Join(dir, "x.go")
	err = ioutil.WriteFile(src, []byte(issue16214src), 0644)
	if err != nil {
		t.Fatalf("could not write file: %v", err)
	}

	cmd := exec.Command(testenv.GoToolPath(t), "tool", "compile", "-S", "-o", filepath.Join(dir, "out.o"), src)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("fail to run go tool compile: %v", err)
	}

	if strings.Contains(string(out), "unknown line number") {
		t.Errorf("line number missing in assembly:\n%s", out)
	}
}

var issue16214src = `
package main

func Mod32(x uint32) uint32 {
	return x % 3 // frontend rewrites it as HMUL with 2863311531, the LITERAL node has Lineno 0
}
`
