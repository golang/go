// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import _ "unsafe"

// As of Go 1.22, the symbols below are found to be pulled via
// linkname in the wild. We provide a push linkname here, to
// keep them accessible with pull linknames.
// This may change in the future. Please do not depend on them
// in new code.

// These should be an internal details
// but widely used packages access them using linkname.
// Do not remove or change the type signature.
// See go.dev/issue/67401.

//go:linkname fastexprand
//go:linkname gopanic
//go:linkname sched
//go:linkname startTheWorld
//go:linkname stopTheWorld
//go:linkname stringHash

// Notable members of the hall of shame include:
//   - github.com/dgraph-io/ristretto
//   - github.com/outcaste-io/ristretto
//   - github.com/clubpay/ronykit
//go:linkname cputicks
