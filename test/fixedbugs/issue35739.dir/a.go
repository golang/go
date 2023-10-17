// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

type myError string

func (e myError) Error() string { return string(e) }

const myErrorVal myError = "error"

func IsMyError(err error) bool {
	return err == error(myErrorVal)
}
