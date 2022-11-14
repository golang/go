// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Export guts for testing.

package runtime

import (
	"runtime/internal/syscall"
)

const SiginfoMaxSize = _si_max_size
const SigeventMaxSize = _sigev_max_size

var Closeonexec = syscall.CloseOnExec
var NewOSProc0 = newosproc0
var Mincore = mincore
var Add = add

type Siginfo siginfo
type Sigevent sigevent
