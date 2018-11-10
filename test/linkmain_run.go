// +build !nacl
// run

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Run the sinit test.

package main

import (
	"fmt"
	"os"
	"os/exec"
	"strings"
)

func cleanup() {
	os.Remove("linkmain.o")
	os.Remove("linkmain.a")
	os.Remove("linkmain1.o")
	os.Remove("linkmain1.a")
	os.Remove("linkmain.exe")
}

func run(cmdline string) {
	args := strings.Fields(cmdline)
	cmd := exec.Command(args[0], args[1:]...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		fmt.Printf("$ %s\n", cmdline)
		fmt.Println(string(out))
		fmt.Println(err)
		cleanup()
		os.Exit(1)
	}
}

func runFail(cmdline string) {
	args := strings.Fields(cmdline)
	cmd := exec.Command(args[0], args[1:]...)
	out, err := cmd.CombinedOutput()
	if err == nil {
		fmt.Printf("$ %s\n", cmdline)
		fmt.Println(string(out))
		fmt.Println("SHOULD HAVE FAILED!")
		cleanup()
		os.Exit(1)
	}
}

func main() {
	// helloworld.go is package main
	run("go tool compile -o linkmain.o helloworld.go")
	run("go tool compile -pack -o linkmain.a helloworld.go")
	run("go tool link -o linkmain.exe linkmain.o")
	run("go tool link -o linkmain.exe linkmain.a")

	// linkmain.go is not
	run("go tool compile -o linkmain1.o linkmain.go")
	run("go tool compile -pack -o linkmain1.a linkmain.go")
	runFail("go tool link -o linkmain.exe linkmain1.o")
	runFail("go tool link -o linkmain.exe linkmain1.a")
	cleanup()
}
