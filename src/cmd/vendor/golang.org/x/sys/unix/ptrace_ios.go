// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ios

package unix

func ptrace(request int, pid int, addr uintptr, data uintptr) (err error) {
	return ENOTSUP
}
