// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && !(mips || mipsle || mips64 || mips64le)

package unix

type siErrnoCode struct {
	Errno int32
	Code  int32
}
