// build -goexperiment arenas

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"arena"
	"log"
	"reflect"
)

func main() {
	a := arena.NewArena()
	defer a.Free()

	const iValue = 10

	i := arena.New[int](a)
	*i = iValue

	if *i != iValue {
		// This test doesn't reasonably expect this to fail. It's more likely
		// that *i crashes for some reason. Still, why not check it.
		log.Fatalf("bad i value: got %d, want %d", *i, iValue)
	}

	const wantLen = 125
	const wantCap = 1912

	sl := arena.MakeSlice[*int](a, wantLen, wantCap)
	if len(sl) != wantLen {
		log.Fatalf("bad arena slice length: got %d, want %d", len(sl), wantLen)
	}
	if cap(sl) != wantCap {
		log.Fatalf("bad arena slice capacity: got %d, want %d", cap(sl), wantCap)
	}
	sl = sl[:cap(sl)]
	for j := range sl {
		sl[j] = i
	}
	for j := range sl {
		if *sl[j] != iValue {
			// This test doesn't reasonably expect this to fail. It's more likely
			// that sl[j] crashes for some reason. Still, why not check it.
			log.Fatalf("bad sl[j] value: got %d, want %d", *sl[j], iValue)
		}
	}

	t := reflect.TypeOf(int(0))
	v := reflect.ArenaNew(a, t)
	if want := reflect.PointerTo(t); v.Type() != want {
		log.Fatalf("unexpected type for arena-allocated value: got %s, want %s", v.Type(), want)
	}
	i2 := v.Interface().(*int)
	*i2 = iValue

	if *i2 != iValue {
		// This test doesn't reasonably expect this to fail. It's more likely
		// that *i crashes for some reason. Still, why not check it.
		log.Fatalf("bad i2 value: got %d, want %d", *i2, iValue)
	}
}
