// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strings

func (r *Replacer) Replacer() interface{} {
	return r.r
}
