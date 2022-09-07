// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && (386 || amd64 || mips || mipsle || mips64 || mipsle || ppc64 || ppc64le || ppc || s390x || sparc64)
// +build linux
// +build 386 amd64 mips mipsle mips64 mipsle ppc64 ppc64le ppc s390x sparc64

package unix

// SYS_ALARM is not defined on arm or riscv, but is available for other GOARCH
// values.

//sys	Alarm(seconds uint) (remaining uint, err error)
