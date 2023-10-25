// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Export guts for testing.

package runtime

const SiginfoMaxSize = _si_max_size
const SigeventMaxSize = _sigev_max_size

var NewOSProc0 = newosproc0
var Mincore = mincore

type Siginfo siginfo
type Sigevent sigevent
