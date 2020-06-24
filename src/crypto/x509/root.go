// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x509

//go:generate go run root_darwin_ios_gen.go -version 55161.80.1

import "sync"

var (
	once           sync.Once
	systemRoots    *CertPool
	systemRootsErr error
)

func systemRootsPool() *CertPool {
	once.Do(initSystemRoots)
	return systemRoots
}

func initSystemRoots() {
	systemRoots, systemRootsErr = loadSystemRoots()
	if systemRootsErr != nil {
		systemRoots = nil
	}
}
