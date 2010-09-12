// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The exec package runs external commands.
package exec

import (
	"os"
)

// Arguments to Run.
const (
	DevNull = iota
	PassThrough
	Pipe
	MergeWithStdout
)

// A Cmd represents a running command.
// Stdin, Stdout, and Stderr are Files representing pipes
// connected to the running command's standard input, output, and error,
// or else nil, depending on the arguments to Run.
// Pid is the running command's operating system process ID.
type Cmd struct {
	Stdin  *os.File
	Stdout *os.File
	Stderr *os.File
	Pid    int
}

// Given mode (DevNull, etc), return file for child
// and file to record in Cmd structure.
func modeToFiles(mode, fd int) (*os.File, *os.File, os.Error) {
	switch mode {
	case DevNull:
		rw := os.O_WRONLY
		if fd == 0 {
			rw = os.O_RDONLY
		}
		f, err := os.Open(os.DevNull, rw, 0)
		return f, nil, err
	case PassThrough:
		switch fd {
		case 0:
			return os.Stdin, nil, nil
		case 1:
			return os.Stdout, nil, nil
		case 2:
			return os.Stderr, nil, nil
		}
	case Pipe:
		r, w, err := os.Pipe()
		if err != nil {
			return nil, nil, err
		}
		if fd == 0 {
			return r, w, nil
		}
		return w, r, nil
	}
	return nil, nil, os.EINVAL
}

// Run starts the named binary running with
// arguments argv and environment envv.
// It returns a pointer to a new Cmd representing
// the command or an error.
//
// The parameters stdin, stdout, and stderr
// specify how to handle standard input, output, and error.
// The choices are DevNull (connect to /dev/null),
// PassThrough (connect to the current process's standard stream),
// Pipe (connect to an operating system pipe), and
// MergeWithStdout (only for standard error; use the same
// file descriptor as was used for standard output).
// If a parameter is Pipe, then the corresponding field (Stdin, Stdout, Stderr)
// of the returned Cmd is the other end of the pipe.
// Otherwise the field in Cmd is nil.
func Run(name string, argv, envv []string, dir string, stdin, stdout, stderr int) (p *Cmd, err os.Error) {
	p = new(Cmd)
	var fd [3]*os.File

	if fd[0], p.Stdin, err = modeToFiles(stdin, 0); err != nil {
		goto Error
	}
	if fd[1], p.Stdout, err = modeToFiles(stdout, 1); err != nil {
		goto Error
	}
	if stderr == MergeWithStdout {
		fd[2] = fd[1]
	} else if fd[2], p.Stderr, err = modeToFiles(stderr, 2); err != nil {
		goto Error
	}

	// Run command.
	p.Pid, err = os.ForkExec(name, argv, envv, dir, fd[0:])
	if err != nil {
		goto Error
	}
	if fd[0] != os.Stdin {
		fd[0].Close()
	}
	if fd[1] != os.Stdout {
		fd[1].Close()
	}
	if fd[2] != os.Stderr && fd[2] != fd[1] {
		fd[2].Close()
	}
	return p, nil

Error:
	if fd[0] != os.Stdin && fd[0] != nil {
		fd[0].Close()
	}
	if fd[1] != os.Stdout && fd[1] != nil {
		fd[1].Close()
	}
	if fd[2] != os.Stderr && fd[2] != nil && fd[2] != fd[1] {
		fd[2].Close()
	}
	if p.Stdin != nil {
		p.Stdin.Close()
	}
	if p.Stdout != nil {
		p.Stdout.Close()
	}
	if p.Stderr != nil {
		p.Stderr.Close()
	}
	return nil, err
}

// Wait waits for the running command p,
// returning the Waitmsg returned by os.Wait and an error.
// The options are passed through to os.Wait.
// Setting options to 0 waits for p to exit;
// other options cause Wait to return for other
// process events; see package os for details.
func (p *Cmd) Wait(options int) (*os.Waitmsg, os.Error) {
	if p.Pid <= 0 {
		return nil, os.ErrorString("exec: invalid use of Cmd.Wait")
	}
	w, err := os.Wait(p.Pid, options)
	if w != nil && (w.Exited() || w.Signaled()) {
		p.Pid = -1
	}
	return w, err
}

// Close waits for the running command p to exit,
// if it hasn't already, and then closes the non-nil file descriptors
// p.Stdin, p.Stdout, and p.Stderr.
func (p *Cmd) Close() os.Error {
	if p.Pid > 0 {
		// Loop on interrupt, but
		// ignore other errors -- maybe
		// caller has already waited for pid.
		_, err := p.Wait(0)
		for err == os.EINTR {
			_, err = p.Wait(0)
		}
	}

	// Close the FDs that are still open.
	var err os.Error
	if p.Stdin != nil && p.Stdin.Fd() >= 0 {
		if err1 := p.Stdin.Close(); err1 != nil {
			err = err1
		}
	}
	if p.Stdout != nil && p.Stdout.Fd() >= 0 {
		if err1 := p.Stdout.Close(); err1 != nil && err != nil {
			err = err1
		}
	}
	if p.Stderr != nil && p.Stderr != p.Stdout && p.Stderr.Fd() >= 0 {
		if err1 := p.Stderr.Close(); err1 != nil && err != nil {
			err = err1
		}
	}
	return err
}
