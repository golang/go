// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Check that there is only one error (no follow-on errors).

package p
// TODO(rFindley) This is a parser error, but in types2 it is a type checking
//                error. We could probably do without this check in the parser.
var _ = [... /* ERROR expected array length, found '...' */ ]byte("foo")
