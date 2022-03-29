// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func f[_ comparable]() {}
func g[_ interface{interface{comparable; ~int|~string}}]() {}

func _[P comparable,
        Q interface{ comparable; ~int|~string },
        R any,                               // not comparable
        S interface{ comparable; ~func() },  // not comparable
]() {
        _ = f[int]
        _ = f[P]
        _ = f[Q]
        _ = f[func /* ERROR does not implement comparable */ ()]
        _ = f[R /* ERROR R does not implement comparable */ ]

        _ = g[int]
        _ = g[P /* ERROR P does not implement interface{interface{comparable; ~int\|~string} */ ]
        _ = g[Q]
        _ = g[func /* ERROR func\(\) does not implement interface{interface{comparable; ~int\|~string}} */ ()]
        _ = g[R /* ERROR R does not implement interface{interface{comparable; ~int\|~string} */ ]
}
