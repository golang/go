// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build extdep

package main

import (
	"flag"
	"fmt"
	"io"
	"os"
)

// put a .tar.gz
func putTar(args []string) error {
	fs := flag.NewFlagSet("put", flag.ContinueOnError)
	fs.Usage = func() {
		fmt.Fprintln(os.Stderr, "create usage: gomote puttar [put-opts] <buildlet-name> [tar.gz file or '-' for stdin]\n")
		fs.PrintDefaults()
		os.Exit(1)
	}
	var rev string
	fs.StringVar(&rev, "gorev", "", "If non-empty, git hash to download from gerrit and put to the buildlet. e.g. 886b02d705ff for Go 1.4.1. This just maps to the --URL flag, so the two options are mutually exclusive.")
	var dir string
	fs.StringVar(&dir, "dir", "", "relative directory from buildlet's work dir to extra tarball into")
	var tarURL string
	fs.StringVar(&tarURL, "url", "", "URL of tarball, instead of provided file.")

	fs.Parse(args)
	if fs.NArg() < 1 || fs.NArg() > 2 {
		fs.Usage()
	}
	if rev != "" {
		if tarURL != "" {
			fmt.Fprintln(os.Stderr, "--gorev and --url are mutually exclusive")
			fs.Usage()
		}
		tarURL = "https://go.googlesource.com/go/+archive/" + rev + ".tar.gz"
	}

	name := fs.Arg(0)
	bc, err := namedClient(name)
	if err != nil {
		return err
	}

	if tarURL != "" {
		if fs.NArg() != 1 {
			fs.Usage()
		}
		return bc.PutTarFromURL(tarURL, dir)
	}

	var tgz io.Reader = os.Stdin
	if fs.NArg() == 2 && fs.Arg(1) != "-" {
		f, err := os.Open(fs.Arg(1))
		if err != nil {
			return err
		}
		defer f.Close()
		tgz = f
	}
	return bc.PutTar(tgz, dir)
}

// put single files
func put(args []string) error {
	fs := flag.NewFlagSet("put", flag.ContinueOnError)
	fs.Usage = func() {
		fmt.Fprintln(os.Stderr, "create usage: gomote put [put-opts] <type>\n\n")
		fs.PrintDefaults()
		os.Exit(1)
	}
	fs.Parse(args)
	if fs.NArg() != 1 {
		fs.Usage()
	}
	return fmt.Errorf("TODO")
	builderType := fs.Arg(0)
	_ = builderType
	return nil
}
