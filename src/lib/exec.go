// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package exec

import (
	"os";
	"syscall";
)

const (
	DevNull = iota;
	Passthru;
	Pipe;
	MergeWithStdout;
)

type Cmd struct {
	Stdin *os.FD;
	Stdout *os.FD;
	Stderr *os.FD;
	Pid int;
}

// Given mode (DevNull, etc), return fd for child
// and fd to record in Cmd structure.
func modeToFDs(mode, fd int) (*os.FD, *os.FD, *os.Error) {
	switch mode {
	case DevNull:
		rw := os.O_WRONLY;
		if fd == 0 {
			rw = os.O_RDONLY;
		}
		f, err := os.Open("/dev/null", rw, 0);
		return f, nil, err;
	case Passthru:
		switch fd {
		case 0:
			return os.Stdin, nil, nil;
		case 1:
			return os.Stdout, nil, nil;
		case 2:
			return os.Stderr, nil, nil;
		}
	case Pipe:
		r, w, err := os.Pipe();
		if err != nil {
			return nil, nil, err;
		}
		if fd == 0 {
			return r, w, nil;
		}
		return w, r, nil;
	}
	return nil, nil, os.EINVAL;
}

// Start command running with pipes possibly
// connected to stdin, stdout, stderr.
// TODO(rsc): Should the stdin,stdout,stderr args
// be [3]int instead?
func OpenCmd(argv0 string, argv, envv []string, stdin, stdout, stderr int)
	(p *Cmd, err *os.Error)
{
	p = new(Cmd);
	var fd [3]*os.FD;

	if fd[0], p.Stdin, err = modeToFDs(stdin, 0); err != nil {
		goto Error;
	}
	if fd[1], p.Stdout, err = modeToFDs(stdout, 1); err != nil {
		goto Error;
	}
	if stderr == MergeWithStdout {
		p.Stderr = p.Stdout;
	} else if fd[2], p.Stderr, err = modeToFDs(stderr, 2); err != nil {
		goto Error;
	}

	// Run command.
	p.Pid, err = os.ForkExec(argv0, argv, envv, fd);
	if err != nil {
		goto Error;
	}
	if fd[0] != os.Stdin {
		fd[0].Close();
	}
	if fd[1] != os.Stdout {
		fd[1].Close();
	}
	if fd[2] != os.Stderr && fd[2] != fd[1] {
		fd[2].Close();
	}
	return p, nil;

Error:
	if fd[0] != os.Stdin && fd[0] != nil {
		fd[0].Close();
	}
	if fd[1] != os.Stdout && fd[1] != nil {
		fd[1].Close();
	}
	if fd[2] != os.Stderr && fd[2] != nil && fd[2] != fd[1] {
		fd[2].Close();
	}
	if p.Stdin != nil {
		p.Stdin.Close();
	}
	if p.Stdout != nil {
		p.Stdout.Close();
	}
	if p.Stderr != nil {
		p.Stderr.Close();
	}
	return nil, err;
}

func (p *Cmd) Wait(options uint64) (*os.Waitmsg, *os.Error) {
	if p.Pid < 0 {
		return nil, os.EINVAL;
	}
	w, err := os.Wait(p.Pid, options);
	if w != nil && (w.Exited() || w.Signaled()) {
		p.Pid = -1;
	}
	return w, err;
}

func (p *Cmd) Close() *os.Error {
	if p.Pid >= 0 {
		// Loop on interrupt, but
		// ignore other errors -- maybe
		// caller has already waited for pid.
		w, err := p.Wait(0);
		for err == os.EINTR {
			w, err = p.Wait(0);
		}
	}

	// Close the FDs that are still open.
	var err *os.Error;
	if p.Stdin != nil && p.Stdin.Fd() >= 0 {
		if err1 := p.Stdin.Close(); err1 != nil {
			err = err1;
		}
	}
	if p.Stdout != nil && p.Stdout.Fd() >= 0 {
		if err1 := p.Stdout.Close(); err1 != nil && err != nil {
			err = err1;
		}
	}
	if p.Stderr != nil && p.Stderr != p.Stdout && p.Stderr.Fd() >= 0 {
		if err1 := p.Stderr.Close(); err1 != nil && err != nil {
			err = err1;
		}
	}
	return err;
}

