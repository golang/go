// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build amd64 amd64p32 386

package runtime

// stackcheck checks that SP is in range [g->stack.lo, g->stack.hi).
func stackcheck()
