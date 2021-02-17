// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x509

// To update the embedded iOS root store, update the -version
// argument to the latest security_certificates version from
// https://opensource.apple.com/source/security_certificates/
// and run "go generate". See https://golang.org/issue/38843.
//go:generate go run root_ios_gen.go -version 55188.40.9

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
