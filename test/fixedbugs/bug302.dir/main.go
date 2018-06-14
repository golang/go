// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// Check that the export information is correct in p.6.
import _ "p"

// Check that it's still correct in pp.a (which contains p.6).
import _ "pp"

