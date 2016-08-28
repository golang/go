// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"net/http"
	"os/exec"
	"runtime"
	"strings"
)

var cmdBug = &Command{
	Run:       runBug,
	UsageLine: "bug",
	Short:     "print information for bug reports",
	Long: `
Bug prints information that helps file effective bug reports.

Bugs may be reported at https://golang.org/issue/new.
	`,
}

func init() {
	cmdBug.Flag.BoolVar(&buildV, "v", false, "")
}

func runBug(cmd *Command, args []string) {
	inspectGoVersion()
	fmt.Println("```")
	fmt.Printf("go version %s %s/%s\n", runtime.Version(), runtime.GOOS, runtime.GOARCH)
	for _, e := range mkEnv() {
		fmt.Printf("%s=\"%s\"\n", e.name, e.value)
	}
	printOSDetails()
	printCDetails()
	fmt.Println("```")
}

func printOSDetails() {
	switch runtime.GOOS {
	case "darwin":
		printCmdOut("uname -v: ", "uname", "-v")
		printCmdOut("", "sw_vers")
	case "linux":
		printCmdOut("uname -sr: ", "uname", "-sr")
		printCmdOut("libc:", "/lib/libc.so.6")
	case "openbsd", "netbsd", "freebsd", "dragonfly":
		printCmdOut("uname -v: ", "uname", "-v")
	case "solaris":
		out, err := ioutil.ReadFile("/etc/release")
		if err == nil {
			fmt.Printf("/etc/release: %s\n", out)
		} else {
			if buildV {
				fmt.Printf("failed to read /etc/release: %v\n", err)
			}
		}
	}
}

func printCDetails() {
	printCmdOut("lldb --version: ", "lldb", "--version")
	cmd := exec.Command("gdb", "--version")
	out, err := cmd.Output()
	if err == nil {
		// There's apparently no combination of command line flags
		// to get gdb to spit out its version without the license and warranty.
		// Print up to the first newline.
		idx := bytes.Index(out, []byte{'\n'})
		line := out[:idx]
		line = bytes.TrimSpace(line)
		fmt.Printf("gdb --version: %s\n", line)
	} else {
		if buildV {
			fmt.Printf("failed to run gdb --version: %v\n", err)
		}
	}
}

func inspectGoVersion() {
	resp, err := http.Get("https://golang.org/VERSION?m=text")
	if err != nil {
		if buildV {
			fmt.Printf("failed to GET golang.org/VERSION: %v\n", err)
		}
		return
	}
	defer resp.Body.Close()
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		if buildV {
			fmt.Printf("failed to read from golang.org/VERSION: %v\n", err)
		}
		return
	}

	// golang.org/VERSION currently returns a whitespace-free string,
	// but just in case, protect against that changing.
	// Similarly so for runtime.Version.
	release := string(bytes.TrimSpace(body))
	vers := strings.TrimSpace(runtime.Version())

	if vers == release {
		// Up to date
		return
	}

	// Devel version or outdated release. Either way, this request is apropos.
	fmt.Printf("Please check whether the issue also reproduces on the latest release, %s.\n\n", release)
}

// printCmdOut prints the output of running the given command.
// It ignores failures; 'go bug' is best effort.
func printCmdOut(prefix, path string, args ...string) {
	cmd := exec.Command(path, args...)
	out, err := cmd.Output()
	if err != nil {
		if buildV {
			fmt.Printf("%s %s: %v\n", path, strings.Join(args, " "), err)
		}
		return
	}
	fmt.Printf("%s%s\n", prefix, bytes.TrimSpace(out))
}
