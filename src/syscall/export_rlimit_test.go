// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

package syscall

func OrigRlimitNofile() Rlimit {
	if rlim := origRlimitNofile.Load(); rlim != nil {
		return *rlim
	}
	return Rlimit{0, 0}
}
