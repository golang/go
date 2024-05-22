// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Declarations for operating systems implementing time.now
// indirectly, in terms of walltime and nanotime assembly.

//go:build !faketime && !windows && !(linux && amd64)

package runtime

import _ "unsafe" // for go:linkname

// time_now should be an internal detail,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - gitee.com/quant1x/gox
//   - github.com/sethvargo/go-limiter
//   - github.com/ulule/limiter/v3
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname time_now time.now
func time_now() (sec int64, nsec int32, mono int64) {
	sec, nsec = walltime()
	return sec, nsec, nanotime()
}
