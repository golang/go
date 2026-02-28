// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import _ "unsafe"

// These should be an internal details
// but widely used packages access them using linkname.
// Do not remove or change the type signature.
// See go.dev/issue/67401.

// Notable members of the hall of shame include:
//   - github.com/dgraph-io/ristretto
//   - github.com/outcaste-io/ristretto
//   - github.com/clubpay/ronykit
//go:linkname cputicks

// Notable members of the hall of shame include:
//   - gvisor.dev/gvisor (from assembly)
//go:linkname sched
