// run

// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 4748.
// This program used to complain because inlining created two exit labels.

package main

func jump() {
        goto exit
exit:
        return
}
func main() {
        jump()
        jump()
}
