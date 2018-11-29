// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package objabi

import (
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"os"
	"strconv"
	"strings"
)

func Flagcount(name, usage string, val *int) {
	flag.Var((*count)(val), name, usage)
}

func Flagfn1(name, usage string, f func(string)) {
	flag.Var(fn1(f), name, usage)
}

func Flagprint(w io.Writer) {
	flag.CommandLine.SetOutput(w)
	flag.PrintDefaults()
}

func Flagparse(usage func()) {
	flag.Usage = usage
	os.Args = expandArgs(os.Args)
	flag.Parse()
}

// expandArgs expands "response files" arguments in the provided slice.
//
// A "response file" argument starts with '@' and the rest of that
// argument is a filename with CR-or-CRLF-separated arguments. Each
// argument in the named files can also contain response file
// arguments. See Issue 18468.
//
// The returned slice 'out' aliases 'in' iff the input did not contain
// any response file arguments.
//
// TODO: handle relative paths of recursive expansions in different directories?
// Is there a spec for this? Are relative paths allowed?
func expandArgs(in []string) (out []string) {
	// out is nil until we see a "@" argument.
	for i, s := range in {
		if strings.HasPrefix(s, "@") {
			if out == nil {
				out = make([]string, 0, len(in)*2)
				out = append(out, in[:i]...)
			}
			slurp, err := ioutil.ReadFile(s[1:])
			if err != nil {
				log.Fatal(err)
			}
			args := strings.Split(strings.TrimSpace(strings.Replace(string(slurp), "\r", "", -1)), "\n")
			out = append(out, expandArgs(args)...)
		} else if out != nil {
			out = append(out, s)
		}
	}
	if out == nil {
		return in
	}
	return
}

func AddVersionFlag() {
	flag.Var(versionFlag{}, "V", "print version and exit")
}

var buildID string // filled in by linker

type versionFlag struct{}

func (versionFlag) IsBoolFlag() bool { return true }
func (versionFlag) Get() interface{} { return nil }
func (versionFlag) String() string   { return "" }
func (versionFlag) Set(s string) error {
	name := os.Args[0]
	name = name[strings.LastIndex(name, `/`)+1:]
	name = name[strings.LastIndex(name, `\`)+1:]
	name = strings.TrimSuffix(name, ".exe")
	p := Expstring()
	if p == DefaultExpstring() {
		p = ""
	}
	sep := ""
	if p != "" {
		sep = " "
	}

	// The go command invokes -V=full to get a unique identifier
	// for this tool. It is assumed that the release version is sufficient
	// for releases, but during development we include the full
	// build ID of the binary, so that if the compiler is changed and
	// rebuilt, we notice and rebuild all packages.
	if s == "full" {
		// If there's an active experiment, include that,
		// to distinguish go1.10.2 with an experiment
		// from go1.10.2 without an experiment.
		if x := Expstring(); x != "" {
			p += " " + x
		}
		if strings.HasPrefix(Version, "devel") {
			p += " buildID=" + buildID
		}
	}

	fmt.Printf("%s version %s%s%s\n", name, Version, sep, p)
	os.Exit(0)
	return nil
}

// count is a flag.Value that is like a flag.Bool and a flag.Int.
// If used as -name, it increments the count, but -name=x sets the count.
// Used for verbose flag -v.
type count int

func (c *count) String() string {
	return fmt.Sprint(int(*c))
}

func (c *count) Set(s string) error {
	switch s {
	case "true":
		*c++
	case "false":
		*c = 0
	default:
		n, err := strconv.Atoi(s)
		if err != nil {
			return fmt.Errorf("invalid count %q", s)
		}
		*c = count(n)
	}
	return nil
}

func (c *count) Get() interface{} {
	return int(*c)
}

func (c *count) IsBoolFlag() bool {
	return true
}

func (c *count) IsCountFlag() bool {
	return true
}

type fn1 func(string)

func (f fn1) Set(s string) error {
	f(s)
	return nil
}

func (f fn1) String() string { return "" }
