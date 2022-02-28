// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build cgo && !netgo

package net

type addrinfoErrno int

func (eai addrinfoErrno) Error() string   { return "<nil>" }
func (eai addrinfoErrno) Temporary() bool { return false }
func (eai addrinfoErrno) Timeout() bool   { return false }
