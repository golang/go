// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sha256

import "internal/cpu"

var useAVX2 = cpu.X86.HasAVX2 && cpu.X86.HasBMI2
var useSHA = useAVX2 && cpu.X86.HasSHA
