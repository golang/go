package main

import (
	"bytes"
	"exec"
	"os"
	"strings"
)

// run is a simple wrapper for exec.Run/Close
func run(envv []string, dir string, argv ...string) os.Error {
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

// runLog runs a process and returns the combined stdout/stderr
func runLog(envv []string, dir string, argv ...string) (output string, exitStatus int, err os.Error) {
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
	_, err = b.ReadFrom(p.Stdout)
	if err != nil {
		return
	}
	w, err := p.Wait(0)
	if err != nil {
		return
	}
	return b.String(), w.WaitStatus.ExitStatus(), nil
}

// Find bin in PATH if a relative or absolute path hasn't been specified
func pathLookup(s string) (string, os.Error) {
	if strings.HasPrefix(s, "/") || strings.HasPrefix(s, "./")  || strings.HasPrefix(s, "../") {
		return s, nil
	} 
	return exec.LookPath(s)
}
