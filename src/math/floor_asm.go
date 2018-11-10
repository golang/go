// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build amd64 amd64p32

package math

import "internal/cpu"

var useSSE41 = cpu.X86.HasSSE41
