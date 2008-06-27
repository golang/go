// $G $F.go && $L $F.$A &&./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
  var i uint64 =
    ' ' +
    'a' +
    'ä' +
    '本' +
    '\a' +
    '\b' +
    '\f' +
    '\n' +
    '\r' +
    '\t' +
    '\v' +
    '\\' +
    '\'' +
    '\000' +
    '\123' +
    '\x00' +
    '\xca' +
    '\xFE' +
    '\u0123' +
    '\ubabe' +
    '\U0123ABCD' +
    '\Ucafebabe'
  ;
  if '\Ucafebabe' != 0xcafebabe {
  	print "cafebabe wrong\n";
  	sys.exit(1)
  }
  if i != 0xcc238de1 {
  	print "number is ", i, " should be ", 0xcc238de1, "\n";
  	sys.exit(1)
  }
}
