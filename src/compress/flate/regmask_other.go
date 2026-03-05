// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !amd64

package flate

// shiftMask is a no-op shift mask for non-x86-64.
// The compiler will optimize it away.
const reg8SizeMask64 = 0xff
