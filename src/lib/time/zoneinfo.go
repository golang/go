// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO: Figure out a better place to put this.
package time

// TODO(rsc): Parse zone info file.
// Amazingly, they are portable across OS X, Linux, BSD, Sun, etc.
// I know how, I just don't want to do it right now.

export func LookupTimezone(sec int64) (zone string, offset int, ok bool) {
	return "PDT", -7*60*60, true
}
