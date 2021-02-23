// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

// From DragonFly's <sys/sysctl.h>
const (
	_CTL_KERN           = 1
	_KERN_PROC          = 14
	_KERN_PROC_PATHNAME = 9
)
