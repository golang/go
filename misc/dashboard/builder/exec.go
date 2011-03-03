// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"exec"
	"io"
	"log"
	"os"
	"strings"
)

// run is a simple wrapper for exec.Run/Close
func run(envv []string, dir string, argv ...string) os.Error {
	if *verbose {
		log.Println("run", argv)
	}
	bin, err := pathLookup(argv[0])
	if err != nil {
		return err
	}
	p, err := exec.Run(bin, argv, envv, dir,
		exec.DevNull, exec.DevNull, exec.PassThrough)
	if err != nil {
		return err
	}
	return p.Close()
}

// runLog runs a process and returns the combined stdout/stderr, 
// as well as writing it to logfile (if specified).
func runLog(envv []string, logfile, dir string, argv ...string) (output string, exitStatus int, err os.Error) {
	if *verbose {
		log.Println("runLog", argv)
	}
	bin, err := pathLookup(argv[0])
	if err != nil {
		return
	}
	p, err := exec.Run(bin, argv, envv, dir,
		exec.DevNull, exec.Pipe, exec.MergeWithStdout)
	if err != nil {
		return
	}
	defer p.Close()
	b := new(bytes.Buffer)
	var w io.Writer = b
	if logfile != "" {
		f, err := os.Open(logfile, os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0644)
		if err != nil {
			return
		}
		defer f.Close()
		w = io.MultiWriter(f, b)
	}
	_, err = io.Copy(w, p.Stdout)
	if err != nil {
		return
	}
	wait, err := p.Wait(0)
	if err != nil {
		return
	}
	return b.String(), wait.WaitStatus.ExitStatus(), nil
}

// Find bin in PATH if a relative or absolute path hasn't been specified
func pathLookup(s string) (string, os.Error) {
	if strings.HasPrefix(s, "/") || strings.HasPrefix(s, "./") || strings.HasPrefix(s, "../") {
		return s, nil
	}
	return exec.LookPath(s)
}
