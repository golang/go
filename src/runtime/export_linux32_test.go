// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && (386 || arm || mips || mipsle)

package runtime

type Timespec32 = timespec32

func (ts timespec) DowncastToTimespec32() Timespec32 {
	return ts.mustDowncastToTimespec32()
}
