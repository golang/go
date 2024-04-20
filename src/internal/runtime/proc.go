// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import _ "unsafe"

//go:linkname ProcPin runtime.procPin
//go:nosplit
func ProcPin() int

//go:linkname ProcUnpin runtime.procUnpin
//go:nosplit
func ProcUnpin()
