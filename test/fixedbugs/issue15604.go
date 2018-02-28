// compile

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bug

import "os"

func f(err error) {
	var ok bool
	if err, ok = err.(*os.PathError); ok {
		if err == os.ErrNotExist {
		}
	}
}
