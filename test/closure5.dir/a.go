// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Check correctness of various closure corner cases
// that are expected to be inlined

package a

func f() bool               { return true }
func G() func() func() bool { return func() func() bool { return f } }
