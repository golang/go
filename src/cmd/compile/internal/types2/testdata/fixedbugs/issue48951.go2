// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type (
        A1[P any] [10]A1 /* ERROR illegal cycle */ [P]
        A2[P any] [10]A2 /* ERROR illegal cycle */ [*P]
        A3[P any] [10]*A3[P]

        L1[P any] []L1[P]

        S1[P any] struct{ f S1 /* ERROR illegal cycle */ [P] }
        S2[P any] struct{ f S2 /* ERROR illegal cycle */ [*P] } // like example in issue
        S3[P any] struct{ f *S3[P] }

        I1[P any] interface{ I1 /* ERROR illegal cycle */ [P] }
        I2[P any] interface{ I2 /* ERROR illegal cycle */ [*P] }
        I3[P any] interface{ *I3 /* ERROR interface contains type constraints */ [P] }
)
