// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"internal/testenv"
	"os"
	"path/filepath"
	"strconv"
	"testing"
)

// Issues 43830, 46295
func TestCGOLTO(t *testing.T) {
	testenv.MustHaveCGO(t)
	testenv.MustHaveGoBuild(t)

	t.Parallel()

	goEnv := func(arg string) string {
		cmd := testenv.Command(t, testenv.GoToolPath(t), "env", arg)
		cmd.Stderr = new(bytes.Buffer)

		line, err := cmd.Output()
		if err != nil {
			t.Fatalf("%v: %v\n%s", cmd, err, cmd.Stderr)
		}
		out := string(bytes.TrimSpace(line))
		t.Logf("%v: %q", cmd, out)
		return out
	}

	cc := goEnv("CC")
	cgoCflags := goEnv("CGO_CFLAGS")

	for test := 0; test < 2; test++ {
		t.Run(strconv.Itoa(test), func { t -> testCGOLTO(t, cc, cgoCflags, test) })
	}
}

const test1_main = `
package main

/*
extern int myadd(int, int);
int c_add(int a, int b) {
	return myadd(a, b);
}
*/
import "C"

func main() {
	println(C.c_add(1, 2))
}
`

const test1_add = `
package main

import "C"

/* test */

//export myadd
func myadd(a C.int, b C.int) C.int {
	return a + b
}
`

const test2_main = `
package main

import "fmt"

/*
#include <stdio.h>

void hello(void) {
  printf("hello\n");
}
*/
import "C"

func main() {
	hello := C.hello
	fmt.Printf("%v\n", hello)
}
`

func testCGOLTO(t *testing.T, cc, cgoCflags string, test int) {
	t.Parallel()

	dir := t.TempDir()

	writeTempFile := func(name, contents string) {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(contents), 0644); err != nil {
			t.Fatal(err)
		}
	}

	writeTempFile("go.mod", "module cgolto\n")

	switch test {
	case 0:
		writeTempFile("main.go", test1_main)
		writeTempFile("add.go", test1_add)
	case 1:
		writeTempFile("main.go", test2_main)
	default:
		t.Fatalf("bad case %d", test)
	}

	cmd := testenv.Command(t, testenv.GoToolPath(t), "build")
	cmd.Dir = dir
	cgoCflags += " -flto"
	cmd.Env = append(cmd.Environ(), "CGO_CFLAGS="+cgoCflags)

	t.Logf("CGO_CFLAGS=%q %v", cgoCflags, cmd)
	out, err := cmd.CombinedOutput()
	t.Logf("%s", out)

	if err != nil {
		t.Logf("go build failed: %v", err)

		// Error messages we've seen indicating that LTO is not supported.
		// These errors come from GCC or clang, not Go.
		var noLTO = []string{
			`unrecognized command line option "-flto"`,
			"unable to pass LLVM bit-code files to linker",
			"file not recognized: File format not recognized",
			"LTO support has not been enabled",
			"linker command failed with exit code",
			"gcc: can't load library",
		}
		for _, msg := range noLTO {
			if bytes.Contains(out, []byte(msg)) {
				t.Skipf("C compiler %v does not support LTO", cc)
			}
		}

		t.Error("failed")
	}
}
