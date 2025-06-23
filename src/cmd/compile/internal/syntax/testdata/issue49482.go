// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type (
        // these need a comma to disambiguate
        _[P *T,] struct{}
        _[P *T, _ any] struct{}
        _[P (*T),] struct{}
        _[P (*T), _ any] struct{}
        _[P (T),] struct{}
        _[P (T), _ any] struct{}

        // these parse as name followed by type
        _[P *struct{}] struct{}
        _[P (*struct{})] struct{}
        _[P ([]int)] struct{}

        // array declarations
        _ [P(T)]struct{}
        _ [P((T))]struct{}
        _ [P * *T] struct{} // this could be a name followed by a type but it makes the rules more complicated
        _ [P * T]struct{}
        _ [P(*T)]struct{}
        _ [P(**T)]struct{}
        _ [P * T - T]struct{}
        _ [P*T-T /* ERROR unexpected comma */ ,]struct{}
        _ [10 /* ERROR unexpected comma */ ,]struct{}
)
