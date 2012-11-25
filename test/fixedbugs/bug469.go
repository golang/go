// compile

// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The gccgo compiler would complain about a redefinition of i, but
// the spec imposes no requirements on parameter names in a function
// type.

package p

type F func(i int) (i int)
