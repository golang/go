// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package parse

import (
	"fmt"
	"strconv"
)

// Set returns a slice of Trees created by parsing the template set
// definition in the argument string. If an error is encountered,
// parsing stops and an empty slice is returned with the error.
func Set(text, leftDelim, rightDelim string, funcs ...map[string]interface{}) (tree map[string]*Tree, err error) {
	tree = make(map[string]*Tree)
	defer (*Tree)(nil).recover(&err)
	lex := lex("set", text, leftDelim, rightDelim)
	const context = "define clause"
	for {
		t := New("set") // name will be updated once we know it.
		t.startParse(funcs, lex)
		// Expect EOF or "{{ define name }}".
		if t.atEOF() {
			break
		}
		t.expect(itemLeftDelim, context)
		t.expect(itemDefine, context)
		name := t.expect(itemString, context)
		t.Name, err = strconv.Unquote(name.val)
		if err != nil {
			t.error(err)
		}
		t.expect(itemRightDelim, context)
		end := t.parse(false)
		if end == nil {
			t.errorf("unexpected EOF in %s", context)
		}
		if end.Type() != nodeEnd {
			t.errorf("unexpected %s in %s", end, context)
		}
		t.stopParse()
		if _, present := tree[t.Name]; present {
			return nil, fmt.Errorf("template: %q multiply defined", name)
		}
		tree[t.Name] = t
	}
	return
}
