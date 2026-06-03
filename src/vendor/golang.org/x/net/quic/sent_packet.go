// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package quic

import (
	"sync"
	"time"

	"golang.org/x/net/internal/quic/quicwire"
)

// A sentPacket tracks state related to an in-flight packet we sent,
// to be committed when the peer acks it or resent if the packet is lost.
type sentPacket struct {
	num   packetNumber
	size  int       // size in bytes
	time  time.Time // time sent
	ptype packetType

	state        sentPacketState
	ackEliciting bool // https://www.rfc-editor.org/rfc/rfc9002.html#section-2-3.4.1
	inFlight     bool // https://www.rfc-editor.org/rfc/rfc9002.html#section-2-3.6.1

	// Frames sent in the packet.
	//
	// This is an abbreviated version of the packet payload, containing only the information
	// we need to process an ack for or loss of this packet.
	// For example, a CRYPTO frame is recorded as the frame type (0x06), offset, and length,
	// but does not include the sent data.
	//
	// This buffer is written by packetWriter.append* and read by Conn.handleAckOrLoss.
	b []byte
	n int // read offset into b
}

type sentPacketState uint8

const (
	sentPacketSent   = sentPacketState(iota) // sent but neither acked nor lost
	sentPacketAcked                          // acked
	sentPacketLost                           // declared lost
	sentPacketUnsent                         // never sent
)

var sentPool = sync.Pool{
	New: func() any {
		return &sentPacket{}
	},
}

func newSentPacket() *sentPacket {
	sent := sentPool.Get().(*sentPacket)
	sent.reset()
	return sent
}

// recycle returns a sentPacket to the pool.
func (sent *sentPacket) recycle() {
	sentPool.Put(sent)
}

func (sent *sentPacket) reset() {
	*sent = sentPacket{
		b: sent.b[:0],
	}
}

// markAckEliciting marks the packet as containing an ack-eliciting frame.
func (sent *sentPacket) markAckEliciting() {
	sent.ackEliciting = true
	sent.inFlight = true
}

// The append* methods record information about frames in the packet.

func (sent *sentPacket) appendNonAckElicitingFrame(frameType byte) {
	sent.b = append(sent.b, frameType)
}

func (sent *sentPacket) appendAckElicitingFrame(frameType byte) {
	sent.ackEliciting = true
	sent.inFlight = true
	sent.b = append(sent.b, frameType)
}

func (sent *sentPacket) appendInt(v uint64) {
	sent.b = quicwire.AppendVarint(sent.b, v)
}

func (sent *sentPacket) appendOffAndSize(start int64, size int) {
	sent.b = quicwire.AppendVarint(sent.b, uint64(start))
	sent.b = quicwire.AppendVarint(sent.b, uint64(size))
}

// The next* methods read back information about frames in the packet.

func (sent *sentPacket) next() (frameType byte) {
	f := sent.b[sent.n]
	sent.n++
	return f
}

func (sent *sentPacket) nextInt() uint64 {
	v, n := quicwire.ConsumeVarint(sent.b[sent.n:])
	sent.n += n
	return v
}

func (sent *sentPacket) nextRange() (start, end int64) {
	start = int64(sent.nextInt())
	end = start + int64(sent.nextInt())
	return start, end
}

func (sent *sentPacket) done() bool {
	return sent.n == len(sent.b)
}
