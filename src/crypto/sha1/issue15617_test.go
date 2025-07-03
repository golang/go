// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && (linux || darwin)

package sha1_test

import (
	"crypto/internal/cryptotest"
	"crypto/sha1"
	"syscall"
	"testing"
)

func TestOutOfBoundsRead(t *testing.T) {
	cryptotest.TestAllImplementations(t, "sha1", func(t *testing.T) {
		const pageSize = 4 << 10
		data, err := syscall.Mmap(0, 0, 2*pageSize, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
		if err != nil {
			panic(err)
		}
		if err := syscall.Mprotect(data[pageSize:], syscall.PROT_NONE); err != nil {
			panic(err)
		}
		for i := 0; i < pageSize; i++ {
			sha1.Sum(data[pageSize-i : pageSize])
		}
	})
}
