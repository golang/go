// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64

package math

import "internal/cpu"

var useFMA = cpu.X86.HasAVX && cpu.X86.HasFMA
