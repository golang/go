// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !cmd_go_bootstrap && unix

package work

import (
	"errors"
	"io"
	"os"
	"path/filepath"
	"syscall"

	"golang.org/x/sys/unix"
)

const earlyExportSupported = true

type exportPipe struct {
	r, w *os.File
}

func newExportPipe(objdir string) (*exportPipe, error) {
	path := filepath.Join(objdir, "export.fifo")
	if err := unix.Mkfifo(path, 0600); err != nil {
		return nil, err
	}
	// Without NONBLOCK, open(2) would wait for the writing peer.
	r0, err := os.OpenFile(path, os.O_RDONLY|syscall.O_NONBLOCK, 0)
	if err != nil {
		os.Remove(path)
		return nil, err
	}
	// Keep a write descriptor open in this process so reading from r waits
	// instead of returning immediate EOF before the child opens path.
	w, err := os.OpenFile(path, os.O_WRONLY|syscall.O_NONBLOCK, 0)
	if err != nil {
		r0.Close()
		os.Remove(path)
		return nil, err
	}
	// Open r without O_NONBLOCK now that w is open so os.Open configures
	// polling on platforms that support FIFO polling and blocking mode on
	// Darwin/iOS (issue #24164).
	r, err := os.Open(path)
	r0.Close()
	if err != nil {
		w.Close()
		os.Remove(path)
		return nil, err
	}
	return &exportPipe{
		r: r,
		w: w,
	}, nil
}

func (ep *exportPipe) name() string { return ep.r.Name() }

func (ep *exportPipe) wait() error {
	var b [1]byte
	_, err := ep.r.Read(b[:])
	if errors.Is(err, os.ErrClosed) {
		return io.EOF
	}
	return err
}

func (ep *exportPipe) close() {
	path := ep.r.Name()
	ep.w.Close()
	ep.r.Close()
	os.Remove(path)
}
