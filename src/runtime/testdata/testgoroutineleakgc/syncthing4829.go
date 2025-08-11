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
	mut sync.RWMutex

	extAddresses map[string]Address_syncthing4829
}

func (m *Mapping_syncthing4829) clearAddresses() {
	m.mut.Lock() // First locking
	var removed []Address_syncthing4829
	for id, addr := range m.extAddresses {
		removed = append(removed, addr)
		delete(m.extAddresses, id)
	}
	if len(removed) > 0 {
		m.notify(nil, removed)
	}
	m.mut.Unlock()
}

func (m *Mapping_syncthing4829) notify(added, remove []Address_syncthing4829) {
	m.mut.RLock()
	m.mut.RUnlock()
}

type Service_syncthing4829 struct {
	mut sync.RWMutex

	mappings []*Mapping_syncthing4829
}

func (s *Service_syncthing4829) NewMapping() *Mapping_syncthing4829 {
	mapping := &Mapping_syncthing4829{
		extAddresses: make(map[string]Address_syncthing4829),
	}
	s.mut.Lock()
	s.mappings = append(s.mappings, mapping)
	s.mut.Unlock()
	return mapping
}

func (s *Service_syncthing4829) RemoveMapping(mapping *Mapping_syncthing4829) {
	s.mut.Lock()
	defer s.mut.Unlock()
	for _, existing := range s.mappings {
		if existing == mapping {
			mapping.clearAddresses()
		}
	}
}

func NewService_syncthing4829() *Service_syncthing4829 {
	return &Service_syncthing4829{}
}

func Syncthing4829() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		runtime.Gosched()
		prof.WriteTo(os.Stdout, 2)
	}()

	go func() {
		// deadlocks: 1
		natSvc := NewService_syncthing4829()
		m := natSvc.NewMapping()
		m.extAddresses["test"] = 0

		natSvc.RemoveMapping(m)
	}()
}
