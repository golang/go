// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package exec

import "io/fs"

func init() {
	skipStdinCopyError = func(err error) bool {
		// Ignore hungup errors copying to stdin if the program
		// completed successfully otherwise.
		// See Issue 35753.
		pe, ok := err.(*fs.PathError)
		return ok &&
			pe.Op == "write" && pe.Path == "|1" &&
			pe.Err.Error() == "i/o on hungup channel"
	}
}
