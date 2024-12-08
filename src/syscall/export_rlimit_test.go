// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

package syscall

import "sync/atomic"

func OrigRlimitNofile() *Rlimit {
	return origRlimitNofile.Load()
}

func GetInternalOrigRlimitNofile() *atomic.Pointer[Rlimit] {
	return &origRlimitNofile
}
