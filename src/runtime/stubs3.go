// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !plan9
// +build !solaris
// +build !freebsd
// +build !darwin
// +build !aix

package runtime

func nanotime1() int64
