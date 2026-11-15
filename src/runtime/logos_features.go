// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

// logosRuntimeHooks selects the loGo/s runtime hook operations guarded at
// their call sites. The go command driver may substitute this compile
// input with an equivalent file defining logosRuntimeHooks = false.
const logosRuntimeHooks = true
