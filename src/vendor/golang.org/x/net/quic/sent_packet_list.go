// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package quic

// A sentPacketList is a ring buffer of sentPackets.
//
// Processing an ack for a packet causes all older packets past a small threshold
// to be discarded (RFC 9002, Section 6.1.1), so the list of in-flight packets is
// not sparse and will contain at most a few acked/lost packets we no longer
// care about.
type sentPacketList struct {
	nextNum packetNumber // next packet number to add to the buffer
	off     int          // offset of first packet in the buffer
	size    int          // number of packets
	p       []*sentPacket
}

// start is the first packet in the list.
func (s *sentPacketList) start() packetNumber {
	return s.nextNum - packetNumber(s.size)
}

// end is one after the last packet in the list.
// If the list is empty, start == end.
func (s *sentPacketList) end() packetNumber {
	return s.nextNum
}

// discard clears the list.
func (s *sentPacketList) discard() {
	*s = sentPacketList{}
}

// add appends a packet to the list.
func (s *sentPacketList) add(sent *sentPacket) {
	if s.nextNum != sent.num {
		panic("inserting out-of-order packet")
	}
	s.nextNum++
	if s.size >= len(s.p) {
		s.grow()
	}
	i := (s.off + s.size) % len(s.p)
	s.size++
	s.p[i] = sent
}

// nth returns a packet by index.
func (s *sentPacketList) nth(n int) *sentPacket {
	index := (s.off + n) % len(s.p)
	return s.p[index]
}

// num returns a packet by number.
// It returns nil if the packet is not in the list.
func (s *sentPacketList) num(num packetNumber) *sentPacket {
	i := int(num - s.start())
	if i < 0 || i >= s.size {
		return nil
	}
	return s.nth(i)
}

// clean removes all acked or lost packets from the head of the list.
func (s *sentPacketList) clean() {
	for s.size > 0 {
		sent := s.p[s.off]
		if sent.state == sentPacketSent {
			return
		}
		sent.recycle()
		s.p[s.off] = nil
		s.off = (s.off + 1) % len(s.p)
		s.size--
	}
	s.off = 0
}

// grow increases the buffer to hold more packaets.
func (s *sentPacketList) grow() {
	newSize := len(s.p) * 2
	if newSize == 0 {
		newSize = 64
	}
	p := make([]*sentPacket, newSize)
	for i := 0; i < s.size; i++ {
		p[i] = s.nth(i)
	}
	s.p = p
	s.off = 0
}
