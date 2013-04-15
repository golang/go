// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math/rand"
	"runtime"
	"sync"
)

func mapTypes() []MapType {
	// TODO(bradfitz): bunch more map types of all different key and value types.
	// Use reflect.MapOf and a program to generate lots of types & struct types.
	// For now, just one:
	return []MapType{intMapType{}}
}

type MapType interface {
	NewMap() Map
}

type Map interface {
	AddItem()
	DelItem()
	Len() int
	GetItem()
	RangeAll()
}

func stressMapType(mt MapType, done func()) {
	defer done()
	m := mt.NewMap()
	for m.Len() < 10000 {
		Println("map at ", m.Len())
		if m.Len()%100 == 0 {
			runtime.Gosched()
		}
		m.AddItem()
		m.AddItem()
		m.DelItem()
		var wg sync.WaitGroup
		const numGets = 10
		wg.Add(numGets)
		for i := 0; i < numGets; i++ {
			go func(i int) {
				if i&1 == 0 {
					m.GetItem()
				} else {
					m.RangeAll()
				}
				wg.Done()
			}(i)
		}
		wg.Wait()
	}
	for m.Len() > 0 {
		m.DelItem()
	}
}

type intMapType struct{}

func (intMapType) NewMap() Map {
	return make(intMap)
}

var deadcafe = []byte("\xDE\xAD\xCA\xFE")

type intMap map[int][]byte

func (m intMap) AddItem() {
	s0 := len(m)
	for len(m) == s0 {
		key := rand.Intn(s0 + 1)
		m[key] = make([]byte, rand.Intn(64<<10))
	}
}

func (m intMap) DelItem() {
	for k := range m {
		delete(m, k)
		return
	}
}

func (m intMap) GetItem() {
	key := rand.Intn(len(m))
	if s, ok := m[key]; ok {
		copy(s, deadcafe)
	}
}

func (m intMap) Len() int { return len(m) }

func (m intMap) RangeAll() {
	for _ = range m {
	}
}

func stressMaps() {
	for {
		var wg sync.WaitGroup
		for _, mt := range mapTypes() {
			wg.Add(1)
			go stressMapType(mt, wg.Done)
		}
		wg.Wait()
	}
}
