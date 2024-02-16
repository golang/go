// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !aix && !darwin && !freebsd && !openbsd && !plan9 && !solaris && !wasip1

package runtime

//go:wasmimport gojs runtime.nanotime1
func nanotime1() int64
