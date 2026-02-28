// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unix

import "syscall"

// Reference: https://man.netbsd.org/open.2
const noFollowErrno = syscall.EFTYPE
