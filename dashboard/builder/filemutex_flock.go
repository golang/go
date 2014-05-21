// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd openbsd

package main

import (
	"sync"
	"syscall"
)

// FileMutex is similar to sync.RWMutex, but also synchronizes across processes.
// This implementation is based on flock syscall.
type FileMutex struct {
	mu sync.RWMutex
	fd int
}

func MakeFileMutex(filename string) *FileMutex {
	if filename == "" {
		return &FileMutex{fd: -1}
	}
	fd, err := syscall.Open(filename, syscall.O_CREAT|syscall.O_RDONLY, mkdirPerm)
	if err != nil {
		panic(err)
	}
	return &FileMutex{fd: fd}
}

func (m *FileMutex) Lock() {
	m.mu.Lock()
	if m.fd != -1 {
		if err := syscall.Flock(m.fd, syscall.LOCK_EX); err != nil {
			panic(err)
		}
	}
}

func (m *FileMutex) Unlock() {
	if m.fd != -1 {
		if err := syscall.Flock(m.fd, syscall.LOCK_UN); err != nil {
			panic(err)
		}
	}
	m.mu.Unlock()
}

func (m *FileMutex) RLock() {
	m.mu.RLock()
	if m.fd != -1 {
		if err := syscall.Flock(m.fd, syscall.LOCK_SH); err != nil {
			panic(err)
		}
	}
}

func (m *FileMutex) RUnlock() {
	if m.fd != -1 {
		if err := syscall.Flock(m.fd, syscall.LOCK_UN); err != nil {
			panic(err)
		}
	}
	m.mu.RUnlock()
}
