// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unix

const (
	getrandomTrap       uintptr = 384
	copyFileRangeTrap   uintptr = 391
	pidfdSendSignalTrap uintptr = 424
)
