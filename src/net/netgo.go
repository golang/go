// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Default netGo to true if the netgo build tag is being used, or the
// C library DNS routines are not available. Note that the C library
// routines are always available on Darwin and Windows.

//go:build netgo || (!cgo && !darwin && !windows)

package net

func init() { netGo = true }
