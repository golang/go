// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

func asmstdcall(fn unsafe.Pointer)
func getlasterror() uint32
func setlasterror(err uint32)
func usleep1(usec uint32)
