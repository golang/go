// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x509

import "sync"

var (
	once        sync.Once
	systemRoots *CertPool
)

func systemRootsPool() *CertPool {
	once.Do(initSystemRoots)
	return systemRoots
}
