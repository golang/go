// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

const (
	_SS_DISABLE  = 2
	_SIG_UNBLOCK = 2
	_SIG_SETMASK = 3
	_NSIG        = 73 /* number of signals in sigtable array */
	_SI_USER     = 0
)
