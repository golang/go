// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

// supportsCloseOnExec reports whether the platform supports the
// O_CLOEXEC flag.
// The O_CLOEXEC flag was introduced in OS X 10.7 (Darwin 11.0.0).
// See http://support.apple.com/kb/HT1633.
const supportsCloseOnExec = true
