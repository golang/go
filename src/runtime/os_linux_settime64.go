// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && !(386 || arm || mips || mipsle)

package runtime

//go:noescape
func timer_settime(timerid int32, flags int32, new, old *itimerspec) int32
