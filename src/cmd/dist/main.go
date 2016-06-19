// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"os"
	"strconv"
)

// cmdtab records the available commands.
var cmdtab = []struct {
	name string
	f    func()
}{
	{"banner", cmdbanner},
	{"bootstrap", cmdbootstrap},
	{"clean", cmdclean},
	{"env", cmdenv},
	{"install", cmdinstall},
	{"list", cmdlist},
	{"test", cmdtest},
	{"version", cmdversion},
}

// The OS-specific main calls into the portable code here.
func xmain() {
	if len(os.Args) < 2 {
		usage()
	}
	cmd := os.Args[1]
	os.Args = os.Args[1:] // for flag parsing during cmd
	for _, ct := range cmdtab {
		if ct.name == cmd {
			flag.Usage = func() {
				fmt.Fprintf(os.Stderr, "usage: go tool dist %s [options]\n", cmd)
				flag.PrintDefaults()
				os.Exit(2)
			}
			ct.f()
			return
		}
	}

	xprintf("unknown command %s\n", cmd)
	usage()
}

func xflagparse(maxargs int) {
	flag.Var((*count)(&vflag), "v", "verbosity")
	flag.Parse()
	if maxargs >= 0 && flag.NArg() > maxargs {
		flag.Usage()
	}
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

func (c *count) IsBoolFlag() bool {
	return true
}
