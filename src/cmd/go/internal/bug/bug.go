// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package bug implements the ``go bug'' command.
package bug

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/envcmd"
	"cmd/go/internal/web"
)

var CmdBug = &base.Command{
	Run:       runBug,
	UsageLine: "go bug",
	Short:     "start a bug report",
	Long: `
Bug opens the default browser and starts a new bug report.
The report includes useful system information.
	`,
}

func init() {
	CmdBug.Flag.BoolVar(&cfg.BuildV, "v", false, "")
}

func runBug(cmd *base.Command, args []string) {
	if len(args) > 0 {
		base.Fatalf("go bug: bug takes no arguments")
	}
	var buf bytes.Buffer
	buf.WriteString(bugHeader)
	inspectGoVersion(&buf)
	fmt.Fprint(&buf, "#### System details\n\n")
	fmt.Fprintln(&buf, "```")
	fmt.Fprintf(&buf, "go version %s %s/%s\n", runtime.Version(), runtime.GOOS, runtime.GOARCH)
	env := cfg.CmdEnv
	env = append(env, envcmd.ExtraEnvVars()...)
	for _, e := range env {
		// Hide the TERM environment variable from "go bug".
		// See issue #18128
		if e.Name != "TERM" {
			fmt.Fprintf(&buf, "%s=\"%s\"\n", e.Name, e.Value)
		}
	}
	printGoDetails(&buf)
	printOSDetails(&buf)
	printCDetails(&buf)
	fmt.Fprintln(&buf, "```")

	body := buf.String()
	url := "https://github.com/golang/go/issues/new?body=" + web.QueryEscape(body)
	if !web.OpenBrowser(url) {
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

func printGoDetails(w io.Writer) {
	printCmdOut(w, "GOROOT/bin/go version: ", filepath.Join(runtime.GOROOT(), "bin/go"), "version")
	printCmdOut(w, "GOROOT/bin/go tool compile -V: ", filepath.Join(runtime.GOROOT(), "bin/go"), "tool", "compile", "-V")
}

func printOSDetails(w io.Writer) {
	switch runtime.GOOS {
	case "darwin":
		printCmdOut(w, "uname -v: ", "uname", "-v")
		printCmdOut(w, "", "sw_vers")
	case "linux":
		printCmdOut(w, "uname -sr: ", "uname", "-sr")
		printCmdOut(w, "", "lsb_release", "-a")
		printGlibcVersion(w)
	case "openbsd", "netbsd", "freebsd", "dragonfly":
		printCmdOut(w, "uname -v: ", "uname", "-v")
	case "solaris":
		out, err := ioutil.ReadFile("/etc/release")
		if err == nil {
			fmt.Fprintf(w, "/etc/release: %s\n", out)
		} else {
			if cfg.BuildV {
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
		fmt.Fprintf(w, "gdb --version: %s\n", firstLine(out))
	} else {
		if cfg.BuildV {
			fmt.Printf("failed to run gdb --version: %v\n", err)
		}
	}
}

func inspectGoVersion(w io.Writer) {
	data, err := web.Get("https://golang.org/VERSION?m=text")
	if err != nil {
		if cfg.BuildV {
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
		if cfg.BuildV {
			fmt.Printf("%s %s: %v\n", path, strings.Join(args, " "), err)
		}
		return
	}
	fmt.Fprintf(w, "%s%s\n", prefix, bytes.TrimSpace(out))
}

// firstLine returns the first line of a given byte slice.
func firstLine(buf []byte) []byte {
	idx := bytes.IndexByte(buf, '\n')
	if idx > 0 {
		buf = buf[:idx]
	}
	return bytes.TrimSpace(buf)
}

// printGlibcVersion prints information about the glibc version.
// It ignores failures.
func printGlibcVersion(w io.Writer) {
	tempdir := os.TempDir()
	if tempdir == "" {
		return
	}
	src := []byte(`int main() {}`)
	srcfile := filepath.Join(tempdir, "go-bug.c")
	outfile := filepath.Join(tempdir, "go-bug")
	err := ioutil.WriteFile(srcfile, src, 0644)
	if err != nil {
		return
	}
	defer os.Remove(srcfile)
	cmd := exec.Command("gcc", "-o", outfile, srcfile)
	if _, err = cmd.CombinedOutput(); err != nil {
		return
	}
	defer os.Remove(outfile)

	cmd = exec.Command("ldd", outfile)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return
	}
	re := regexp.MustCompile(`libc\.so[^ ]* => ([^ ]+)`)
	m := re.FindStringSubmatch(string(out))
	if m == nil {
		return
	}
	cmd = exec.Command(m[1])
	out, err = cmd.Output()
	if err != nil {
		return
	}
	fmt.Fprintf(w, "%s: %s\n", m[1], firstLine(out))

	// print another line (the one containing version string) in case of musl libc
	if idx := bytes.IndexByte(out, '\n'); bytes.Index(out, []byte("musl")) != -1 && idx > -1 {
		fmt.Fprintf(w, "%s\n", firstLine(out[idx+1:]))
	}
}
