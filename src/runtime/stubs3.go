// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build plan9

package runtime

func close(fd int32) int32

//go:noescape
func open(name *byte, mode, perm int32) int32
