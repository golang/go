// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package b

import (
	"io/ioutil"

	"./a"
)

var G int

// An inlinable function. To trigger the bug in question this needs
// to be inlined here within the package and also inlined into some
// other package that imports it.
func ReadValues(data []byte) (vals map[string]interface{}, err error) {
	err = a.Unmarshal(data, &vals)
	if len(vals) == 0 {
		vals = map[string]interface{}{}
	}
	return
}

// A local call to the function above, which triggers the "move to heap"
// of the output param.
func CallReadValues(filename string) (map[string]interface{}, error) {
	defer func() { G++ }()
	data, err := ioutil.ReadFile(filename)
	if err != nil {
		return map[string]interface{}{}, err
	}
	return ReadValues(data)
}
