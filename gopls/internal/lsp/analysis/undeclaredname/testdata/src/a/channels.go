// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package undeclared

func channels(s string) {
	undefinedChannels(c()) // want "(undeclared name|undefined): undefinedChannels"
}

func c() (<-chan string, chan string) {
	return make(<-chan string), make(chan string)
}
