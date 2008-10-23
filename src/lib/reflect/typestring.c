// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

extern char gotypestrings[];	// really a go String, but we don't have the definition here

void FLUSH(void *v) { }

void reflect·typestrings(void *s) {
	s = gotypestrings;
	FLUSH(&s);
}
