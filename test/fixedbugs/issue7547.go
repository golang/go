// compile

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func f() map[string]interface{} {
	var p *map[string]map[string]interface{}
	_ = p
	return nil
}

func main() {
	f()
}
