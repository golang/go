// errorcheck

// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 4776: missing package declaration error should be fatal.

type MyInt int32 // ERROR "package statement must be first|package clause"

