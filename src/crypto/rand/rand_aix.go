// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rand

import (
	"errors"
	"io"
	"os"
	"sync"
	"sync/atomic"
	"syscall"
)

const urandomDevice = "/dev/urandom"

var (
	f    io.Reader
	mu   sync.Mutex
	used atomic.Bool
)

func read(b []byte) error {
	if !used.Load() {
		mu.Lock()
		if !used.Load() {
			dev, err := os.Open(urandomDevice)
			if err != nil {
				mu.Unlock()
				return err
			}
			f = hideAgainReader{dev}
			used.Store(true)
		}
		mu.Unlock()
	}
	if _, err := io.ReadFull(f, b); err != nil {
		return err
	}
	return nil
}

// hideAgainReader masks EAGAIN reads from /dev/urandom.
// See golang.org/issue/9205.
type hideAgainReader struct {
	r io.Reader
}

func (hr hideAgainReader) Read(p []byte) (n int, err error) {
	n, err = hr.r.Read(p)
	if errors.Is(err, syscall.EAGAIN) {
		err = nil
	}
	return
}
