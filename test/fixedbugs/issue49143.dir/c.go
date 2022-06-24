// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package c

import "./b"

type Resolver struct{}

type todoResolver struct{ *Resolver }

func (r *todoResolver) F() {
	b.NewLoaders().Loader.Load()
}
