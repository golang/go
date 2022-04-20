// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package b

import (
	"./a"
)

// InteractionRequest is an incoming request Interaction
type InteractionRequest[T a.InteractionDataConstraint] struct {
	a.Interaction[T]
}
