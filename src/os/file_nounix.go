// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !darwin
// +build !dragonfly
// +build !freebsd
// +build !linux
// +build !nacl
// +build !netbsd
// +build !openbsd
// +build !solaris

package os

const supportsMkdirWithSetuidSetgid = true
