// compiledir

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// https://gcc.gnu.org/PR67968

// gccgo compiler crash building the equality and hash functions for a
// type when a return statement requires a conversion to interface
// type of a call of function defined in a different package that
// returns an unnamed type.

package ignored
