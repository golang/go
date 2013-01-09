// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p2

import "runtime"

func E() func() int { return runtime.NumCPU }

func F() func() { return runtime.GC }

func G() func() string { return runtime.GOROOT }

func H() func() { return runtime.Gosched }

func I() func() string { return runtime.Version }
