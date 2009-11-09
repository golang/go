// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*

Goyacc is a version of yacc for Go.
It is written in Go and generates parsers written in Go.

It is largely transliterated from the Inferno version written in Limbo
which in turn was largely transliterated from the Plan 9 version
written in C and documented at

	http://plan9.bell-labs.com/magic/man2html/1/yacc

Yacc adepts will have no trouble adapting to this form of the tool.

The file units.y in this directory is a yacc grammar for a version of
the Unix tool units, also written in Go and largely transliterated
from the Plan 9 C version.

*/
package documentation
