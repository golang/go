// errorcheck

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Offending character % must not be interpreted as
// start of format verb when emitting error message.

package% // ERROR "unexpected %|package name must be an identifier|after package clause|expected declaration"
