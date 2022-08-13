// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func _(s []int) {
        var i, j, k, l int
        _, _, _, _ = i, j, k, l

        for range s {}
        for i = range s {}
        for i, j = range s {}
        for i, j, k /* ERROR range clause permits at most two iteration variables */ = range s {}
        for i, j, k /* ERROR range clause permits at most two iteration variables */, l = range s {}
}

func _(s chan int) {
        var i, j, k, l int
        _, _, _, _ = i, j, k, l

        for range s {}
        for i = range s {}
        for i, j /* ERROR range over .* permits only one iteration variable */ = range s {}
        for i, j /* ERROR range over .* permits only one iteration variable */, k = range s {}
        for i, j /* ERROR range over .* permits only one iteration variable */, k, l = range s {}
}
