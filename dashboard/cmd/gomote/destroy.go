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

func destroy(args []string) error {
	fs := flag.NewFlagSet("destroy", flag.ContinueOnError)
	fs.Usage = func() {
		fmt.Fprintln(os.Stderr, "create usage: gomote destroy <instance>\n\n")
		fs.PrintDefaults()
		os.Exit(1)
	}

	fs.Parse(args)
	if fs.NArg() != 1 {
		fs.Usage()
	}
	name := fs.Arg(0)
	bc, err := namedClient(name)
	if err != nil {
		return err
	}

	// First ask it to kill itself, and then tell GCE to kill it too:
	shutErr := bc.Destroy()
	gceErr := buildlet.DestroyVM(projTokenSource(), *proj, *zone, fmt.Sprintf("mote-%s-%s", username(), name))
	if shutErr != nil {
		return shutErr
	}
	return gceErr
}
