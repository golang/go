// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package parse

// Set returns a slice of Trees created by parsing the template set
// definition in the argument string. If an error is encountered,
// parsing stops and an empty slice is returned with the error.
func Set(text, leftDelim, rightDelim string, funcs ...map[string]interface{}) (tree map[string]*Tree, err error) {
	tree = make(map[string]*Tree)
	// Top-level template name is needed but unused. TODO: clean this up.
	_, err = New("ROOT").Parse(text, leftDelim, rightDelim, tree, funcs...)
	return
}
