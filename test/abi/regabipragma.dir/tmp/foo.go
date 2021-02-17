// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tmp


type S string

//go:noinline
func (s S) FF(t string) string {
        return string(s) + " " + t
}

//go:noinline
//go:registerparams
func F(s,t string) string {
        return s + " " + t
}
