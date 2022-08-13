// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Ensure there is no "imported but not used" error
// if a package wasn't imported in the first place.

package p

import . "/foo" // ERROR could not import \/foo
