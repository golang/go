// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test case for issue 11377: Better synchronization of
// parser after certain syntax errors.

package p

func bad1() {
    if f()) /* ERROR "expected ';', found '\)'" */ {
        return
    }
}

// There shouldn't be any errors down below.

func F1() {}
func F2() {}
func F3() {}
func F4() {}
func F5() {}
func F6() {}
func F7() {}
func F8() {}
func F9() {}
func F10() {}
