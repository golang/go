// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build nacl plan9 solaris

package main

import (
	"log"
	"sync"
)

// FileMutex is similar to sync.RWMutex, but also synchronizes across processes.
// This implementation is a fallback that does not actually provide inter-process synchronization.
type FileMutex struct {
	sync.RWMutex
}

func MakeFileMutex(filename string) *FileMutex {
	return &FileMutex{}
}

func init() {
	log.Printf("WARNING: using fake file mutex." +
		" Don't run more than one of these at once!!!")
}
