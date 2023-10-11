// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.arenas

package reflect

import "arena"

// ArenaNew returns a Value representing a pointer to a new zero value for the
// specified type, allocating storage for it in the provided arena. That is,
// the returned Value's Type is PointerTo(typ).
func ArenaNew(a *arena.Arena, typ Type) Value {
	return ValueOf(arena_New(a, PointerTo(typ)))
}

func arena_New(a *arena.Arena, typ any) any
