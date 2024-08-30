// compiledir

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 36085: gccgo compiler did not generate type descriptor
// for pointer to type alias defined in another package, causing
// linking error.

package ignored
