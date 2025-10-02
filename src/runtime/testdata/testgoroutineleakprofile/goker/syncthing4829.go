// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a MIT
// license that can be found in the LICENSE file.

package main

import (
	"os"
	"runtime"
	"runtime/pprof"
	"sync"
)

func init() {
	register("Syncthing4829", Syncthing4829)
}

type Address_syncthing4829 int

type Mapping_syncthing4829 struct {
	mut sync.RWMutex // L2

	extAddresses map[string]Address_syncthing4829
}

func (m *Mapping_syncthing4829) clearAddresses() {
	m.mut.Lock() // L2
	var removed []Address_syncthing4829
	for id, addr := range m.extAddresses {
		removed = append(removed, addr)
		delete(m.extAddresses, id)
	}
	if len(removed) > 0 {
		m.notify(nil, removed)
	}
	m.mut.Unlock() // L2
}

func (m *Mapping_syncthing4829) notify(added, remove []Address_syncthing4829) {
	m.mut.RLock() // L2
	m.mut.RUnlock() // L2
}

type Service_syncthing4829 struct {
	mut sync.RWMutex // L1

	mappings []*Mapping_syncthing4829
}

func (s *Service_syncthing4829) NewMapping() *Mapping_syncthing4829 {
	mapping := &Mapping_syncthing4829{
		extAddresses: make(map[string]Address_syncthing4829),
	}
	s.mut.Lock() // L1
	s.mappings = append(s.mappings, mapping)
	s.mut.Unlock() // L1
	return mapping
}

func (s *Service_syncthing4829) RemoveMapping(mapping *Mapping_syncthing4829) {
	s.mut.Lock() // L1
	defer s.mut.Unlock() // L1
	for _, existing := range s.mappings {
		if existing == mapping {
			mapping.clearAddresses()
		}
	}
}

func Syncthing4829() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		// Yield several times to allow the child goroutine to run.
		for i := 0; i < yieldCount; i++ {
			runtime.Gosched()
		}
		prof.WriteTo(os.Stdout, 2)
	}()

	go func() { // G1
		natSvc := &Service_syncthing4829{}
		m := natSvc.NewMapping()
		m.extAddresses["test"] = 0

		natSvc.RemoveMapping(m)
	}()
}
