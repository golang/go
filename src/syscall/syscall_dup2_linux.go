// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !android
// +build 386 amd64 arm mips mipsle mips64 mips64le ppc64 ppc64le s390x

package syscall

const _SYS_dup = SYS_DUP2
