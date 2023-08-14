// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type (
        A1[P any] [10]A1 /* ERROR "invalid recursive type" */ [P]
        A2[P any] [10]A2 /* ERROR "invalid recursive type" */ [*P]
        A3[P any] [10]*A3[P]

        L1[P any] []L1[P]

        S1[P any] struct{ f S1 /* ERROR "invalid recursive type" */ [P] }
        S2[P any] struct{ f S2 /* ERROR "invalid recursive type" */ [*P] } // like example in issue
        S3[P any] struct{ f *S3[P] }

        I1[P any] interface{ I1 /* ERROR "invalid recursive type" */ [P] }
        I2[P any] interface{ I2 /* ERROR "invalid recursive type" */ [*P] }
        I3[P any] interface{ *I3 /* ERROR "interface contains type constraints" */ [P] }
)
