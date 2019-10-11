// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !amd64,!s390x,!ppc64le,!arm64

package crc32

func archAvailableIEEE() bool                    { return false }
func archInitIEEE()                              { panic("not available") }
func archUpdateIEEE(crc uint32, p []byte) uint32 { panic("not available") }

func archAvailableCastagnoli() bool                    { return false }
func archInitCastagnoli()                              { panic("not available") }
func archUpdateCastagnoli(crc uint32, p []byte) uint32 { panic("not available") }
