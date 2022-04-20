// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Check that there is only one error (no follow-on errors).

package p
var _ = [ /* ERROR invalid use of .* array */ ...]byte("foo")
