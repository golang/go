// compile

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 2576
package bug

type T struct { a int }

func f(t T) {
        switch _, _ = t.a, t.a; {}
}