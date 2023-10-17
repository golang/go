// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build solaris
// +build solaris

// Package lif provides basic functions for the manipulation of
// logical network interfaces and interface addresses on Solaris.
//
// The package supports Solaris 11 or above.
package lif

import (
	"syscall"
)

type endpoint struct {
	af int
	s  uintptr
}

func (ep *endpoint) close() error {
	return syscall.Close(int(ep.s))
}

func newEndpoints(af int) ([]endpoint, error) {
	var lastErr error
	var eps []endpoint
	afs := []int{syscall.AF_INET, syscall.AF_INET6}
	if af != syscall.AF_UNSPEC {
		afs = []int{af}
	}
	for _, af := range afs {
		s, err := syscall.Socket(af, syscall.SOCK_DGRAM, 0)
		if err != nil {
			lastErr = err
			continue
		}
		eps = append(eps, endpoint{af: af, s: uintptr(s)})
	}
	if len(eps) == 0 {
		return nil, lastErr
	}
	return eps, nil
}
