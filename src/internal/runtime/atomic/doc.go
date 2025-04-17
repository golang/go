// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package atomic provides atomic operations, independent of sync/atomic,
to the runtime.

On most platforms, the compiler is aware of the functions defined
in this package, and they're replaced with platform-specific intrinsics.
On other platforms, generic implementations are made available.

Unless otherwise noted, operations defined in this package are sequentially
consistent across threads with respect to the values they manipulate. More
specifically, operations that happen in a specific order on one thread,
will always be observed to happen in exactly that order by another thread.
*/
package atomic
