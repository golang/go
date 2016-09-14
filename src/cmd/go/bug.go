// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
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
	var buf bytes.Buffer
	buf.WriteString(bugHeader)
	inspectGoVersion(&buf)
	fmt.Fprint(&buf, "#### System details\n\n")
	fmt.Fprintln(&buf, "```")
	fmt.Fprintf(&buf, "go version %s %s/%s\n", runtime.Version(), runtime.GOOS, runtime.GOARCH)
	for _, e := range mkEnv() {
		fmt.Fprintf(&buf, "%s=\"%s\"\n", e.name, e.value)
	}
	printOSDetails(&buf)
	printCDetails(&buf)
	fmt.Fprintln(&buf, "```")

	body := buf.String()
	url := "https://github.com/golang/go/issues/new?body=" + queryEscape(body)
	if !openBrowser(url) {
		fmt.Print("Please file a new issue at golang.org/issue/new using this template:\n\n")
		fmt.Print(body)
	}
}

const bugHeader = `Please answer these questions before submitting your issue. Thanks!

#### What did you do?
If possible, provide a recipe for reproducing the error.
A complete runnable program is good.
A link on play.golang.org is best.


#### What did you expect to see?


#### What did you see instead?


`

func printOSDetails(w io.Writer) {
	switch runtime.GOOS {
	case "darwin":
		printCmdOut(w, "uname -v: ", "uname", "-v")
		printCmdOut(w, "", "sw_vers")
	case "linux":
		printCmdOut(w, "uname -sr: ", "uname", "-sr")
		printCmdOut(w, "libc:", "/lib/libc.so.6")
	case "openbsd", "netbsd", "freebsd", "dragonfly":
		printCmdOut(w, "uname -v: ", "uname", "-v")
	case "solaris":
		out, err := ioutil.ReadFile("/etc/release")
		if err == nil {
			fmt.Fprintf(w, "/etc/release: %s\n", out)
		} else {
			if buildV {
				fmt.Printf("failed to read /etc/release: %v\n", err)
			}
		}
	}
}

func printCDetails(w io.Writer) {
	printCmdOut(w, "lldb --version: ", "lldb", "--version")
	cmd := exec.Command("gdb", "--version")
	out, err := cmd.Output()
	if err == nil {
		// There's apparently no combination of command line flags
		// to get gdb to spit out its version without the license and warranty.
		// Print up to the first newline.
		idx := bytes.Index(out, []byte{'\n'})
		line := out[:idx]
		line = bytes.TrimSpace(line)
		fmt.Fprintf(w, "gdb --version: %s\n", line)
	} else {
		if buildV {
			fmt.Printf("failed to run gdb --version: %v\n", err)
		}
	}
}

func inspectGoVersion(w io.Writer) {
	data, err := httpGET("https://golang.org/VERSION?m=text")
	if err != nil {
		if buildV {
			fmt.Printf("failed to read from golang.org/VERSION: %v\n", err)
		}
		return
	}

	// golang.org/VERSION currently returns a whitespace-free string,
	// but just in case, protect against that changing.
	// Similarly so for runtime.Version.
	release := string(bytes.TrimSpace(data))
	vers := strings.TrimSpace(runtime.Version())

	if vers == release {
		// Up to date
		return
	}

	// Devel version or outdated release. Either way, this request is apropos.
	fmt.Fprintf(w, "#### Does this issue reproduce with the latest release (%s)?\n\n\n", release)
}

// printCmdOut prints the output of running the given command.
// It ignores failures; 'go bug' is best effort.
func printCmdOut(w io.Writer, prefix, path string, args ...string) {
	cmd := exec.Command(path, args...)
	out, err := cmd.Output()
	if err != nil {
		if buildV {
			fmt.Printf("%s %s: %v\n", path, strings.Join(args, " "), err)
		}
		return
	}
	fmt.Fprintf(w, "%s%s\n", prefix, bytes.TrimSpace(out))
}
