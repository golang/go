// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build extdep

package main

import (
	"flag"
	"fmt"
	"os"

	"golang.org/x/tools/dashboard/buildlet"
)

func run(args []string) error {
	fs := flag.NewFlagSet("run", flag.ContinueOnError)
	fs.Usage = func() {
		fmt.Fprintln(os.Stderr, "create usage: gomote run [run-opts] <buildlet-name> <cmd> [args...]")
		fs.PrintDefaults()
		os.Exit(1)
	}

	fs.Parse(args)
	if fs.NArg() < 2 {
		fs.Usage()
	}
	name, cmd := fs.Arg(0), fs.Arg(1)
	bc, err := namedClient(name)
	if err != nil {
		return err
	}

	remoteErr, execErr := bc.Exec(cmd, buildlet.ExecOpts{
		Output: os.Stdout,
	})
	if execErr != nil {
		return fmt.Errorf("Error trying to execute %s: %v", cmd, execErr)
	}
	return remoteErr
}
