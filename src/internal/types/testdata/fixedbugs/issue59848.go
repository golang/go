// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type T struct{}
type I interface{ M() }
var _ I = T /* ERROR "missing method M" */ {} // must not crash
func (T) m() {}
