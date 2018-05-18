// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This is a sample test file illustrating the use
// of error comments with the error test harness.

package p

// The following are invalid error comments; they are
// silently ignored. The prefix must be exactly one of
// "/* ERROR " or "// ERROR ".
//
/*ERROR*/
/*ERROR foo*/
/* ERRORfoo */
/*  ERROR foo */
//ERROR
// ERROR
// ERRORfoo
//  ERROR foo

// This is a valid error comment; it applies to the
// immediately following token. 
import "math" /* ERROR unexpected comma */ ,

// If there are multiple /*-style error comments before
// the next token, only the last one is considered.
type x = /* ERROR ignored */ /* ERROR literal 0 in type declaration */ 0

// A //-style error comment matches any error position
// on the same line.
func () foo() // ERROR method has no receiver
