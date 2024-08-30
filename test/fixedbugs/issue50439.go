// compile

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

var data []struct {
	F string `tag`
}

var V = ([]struct{ F string })(data)
