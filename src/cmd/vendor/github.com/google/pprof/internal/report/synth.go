package report

import (
	"github.com/google/pprof/profile"
)

// synthCode assigns addresses to locations without an address.
type synthCode struct {
	next uint64
	addr map[*profile.Location]uint64 // Synthesized address assigned to a location
}

func newSynthCode(mappings []*profile.Mapping) *synthCode {
	// Find a larger address than any mapping.
	s := &synthCode{next: 1}
	for _, m := range mappings {
		if s.next < m.Limit {
			s.next = m.Limit
		}
	}
	return s
}

// address returns the synthetic address for loc, creating one if needed.
func (s *synthCode) address(loc *profile.Location) uint64 {
	if loc.Address != 0 {
		panic("can only synthesize addresses for locations without an address")
	}
	if addr, ok := s.addr[loc]; ok {
		return addr
	}
	if s.addr == nil {
		s.addr = map[*profile.Location]uint64{}
	}
	addr := s.next
	s.next++
	s.addr[loc] = addr
	return addr
}
