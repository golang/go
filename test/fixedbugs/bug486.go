// compile

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The gccgo lexer had a bug handling nested comments.
// http://gcc.gnu.org/PR61746
// http://code.google.com/p/gofrontend/issues/detail?id=35

package main

/*// comment
*/
