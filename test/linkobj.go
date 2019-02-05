// +build !nacl,!js
// run

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test the compiler -linkobj flag.

package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"strings"
)

var pwd, tmpdir string

func main() {
	dir, err := ioutil.TempDir("", "go-test-linkobj-")
	if err != nil {
		log.Fatal(err)
	}
	pwd, err = os.Getwd()
	if err != nil {
		log.Fatal(err)
	}
	if err := os.Chdir(dir); err != nil {
		os.RemoveAll(dir)
		log.Fatal(err)
	}
	tmpdir = dir

	writeFile("p1.go", `
		package p1
		
		func F() {
			println("hello from p1")
		}
	`)
	writeFile("p2.go", `
		package p2
		
		import "./p1"

		func F() {
			p1.F()
			println("hello from p2")
		}
		
		func main() {}
	`)
	writeFile("p3.go", `
		package main

		import "./p2"
		
		func main() {
			p2.F()
			println("hello from main")
		}
	`)

	// two rounds: once using normal objects, again using .a files (compile -pack).
	for round := 0; round < 2; round++ {
		pkg := "-pack=" + fmt.Sprint(round)

		// The compiler expects the files being read to have the right suffix.
		o := "o"
		if round == 1 {
			o = "a"
		}

		// inlining is disabled to make sure that the link objects contain needed code.
		run("go", "tool", "compile", pkg, "-D", ".", "-I", ".", "-l", "-o", "p1."+o, "-linkobj", "p1.lo", "p1.go")
		run("go", "tool", "compile", pkg, "-D", ".", "-I", ".", "-l", "-o", "p2."+o, "-linkobj", "p2.lo", "p2.go")
		run("go", "tool", "compile", pkg, "-D", ".", "-I", ".", "-l", "-o", "p3."+o, "-linkobj", "p3.lo", "p3.go")

		cp("p1."+o, "p1.oo")
		cp("p2."+o, "p2.oo")
		cp("p3."+o, "p3.oo")
		cp("p1.lo", "p1."+o)
		cp("p2.lo", "p2."+o)
		cp("p3.lo", "p3."+o)
		out := runFail("go", "tool", "link", "p2."+o)
		if !strings.Contains(out, "not package main") {
			fatalf("link p2.o failed but not for package main:\n%s", out)
		}

		run("go", "tool", "link", "-L", ".", "-o", "a.out.exe", "p3."+o)
		out = run("./a.out.exe")
		if !strings.Contains(out, "hello from p1\nhello from p2\nhello from main\n") {
			fatalf("running main, incorrect output:\n%s", out)
		}

		// ensure that mistaken future round can't use these
		os.Remove("p1.o")
		os.Remove("a.out.exe")
	}

	cleanup()
}

func run(args ...string) string {
	out, err := exec.Command(args[0], args[1:]...).CombinedOutput()
	if err != nil {
		fatalf("run %v: %s\n%s", args, err, out)
	}
	return string(out)
}

func runFail(args ...string) string {
	out, err := exec.Command(args[0], args[1:]...).CombinedOutput()
	if err == nil {
		fatalf("runFail %v: unexpected success!\n%s", args, err, out)
	}
	return string(out)
}

func cp(src, dst string) {
	data, err := ioutil.ReadFile(src)
	if err != nil {
		fatalf("%v", err)
	}
	err = ioutil.WriteFile(dst, data, 0666)
	if err != nil {
		fatalf("%v", err)
	}
}

func writeFile(name, data string) {
	err := ioutil.WriteFile(name, []byte(data), 0666)
	if err != nil {
		fatalf("%v", err)
	}
}

func cleanup() {
	const debug = false
	if debug {
		println("TMPDIR:", tmpdir)
		return
	}
	os.Chdir(pwd) // get out of tmpdir before removing it
	os.RemoveAll(tmpdir)
}

func fatalf(format string, args ...interface{}) {
	cleanup()
	log.Fatalf(format, args...)
}
