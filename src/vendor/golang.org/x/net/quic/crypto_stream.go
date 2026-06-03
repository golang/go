// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package quic

// "Implementations MUST support buffering at least 4096 bytes of data
// received in out-of-order CRYPTO frames."
// https://www.rfc-editor.org/rfc/rfc9000.html#section-7.5-2
//
// 4096 is too small for real-world cases, however, so we allow more.
const cryptoBufferSize = 1 << 20

// A cryptoStream is the stream of data passed in CRYPTO frames.
// There is one cryptoStream per packet number space.
type cryptoStream struct {
	// CRYPTO data received from the peer.
	in    pipe
	inset rangeset[int64] // bytes received

	// CRYPTO data queued for transmission to the peer.
	out       pipe
	outunsent rangeset[int64] // bytes in need of sending
	outacked  rangeset[int64] // bytes acked by peer
}

// handleCrypto processes data received in a CRYPTO frame.
func (s *cryptoStream) handleCrypto(off int64, b []byte, f func([]byte) error) error {
	end := off + int64(len(b))
	if end-s.inset.min() > cryptoBufferSize {
		return localTransportError{
			code:   errCryptoBufferExceeded,
			reason: "crypto buffer exceeded",
		}
	}
	s.inset.add(off, end)
	if off == s.in.start {
		// Fast path: This is the next chunk of data in the stream,
		// so just handle it immediately.
		if err := f(b); err != nil {
			return err
		}
		s.in.discardBefore(end)
	} else {
		// This is either data we've already processed,
		// data we can't process yet, or a mix of both.
		s.in.writeAt(b, off)
	}
	// s.in.start is the next byte in sequence.
	// If it's in s.inset, we have bytes to provide.
	// If it isn't, we don't--we're either out of data,
	// or only have data that comes after the next byte.
	if !s.inset.contains(s.in.start) {
		return nil
	}
	// size is the size of the first contiguous chunk of bytes
	// that have not been processed yet.
	size := int(s.inset[0].end - s.in.start)
	if size <= 0 {
		return nil
	}
	err := s.in.read(s.in.start, size, f)
	s.in.discardBefore(s.inset[0].end)
	return err
}

// write queues data for sending to the peer.
// It does not block or limit the amount of buffered data.
// QUIC connections don't communicate the amount of CRYPTO data they are willing to buffer,
// so we send what we have and the peer can close the connection if it is too much.
func (s *cryptoStream) write(b []byte) {
	start := s.out.end
	s.out.writeAt(b, start)
	s.outunsent.add(start, s.out.end)
}

// ackOrLoss reports that an CRYPTO frame sent by us has been acknowledged by the peer, or lost.
func (s *cryptoStream) ackOrLoss(start, end int64, fate packetFate) {
	switch fate {
	case packetAcked:
		s.outacked.add(start, end)
		s.outunsent.sub(start, end)
		// If this ack is for data at the start of the send buffer, we can now discard it.
		if s.outacked.contains(s.out.start) {
			s.out.discardBefore(s.outacked[0].end)
		}
	case packetLost:
		// Mark everything lost, but not previously acked, as needing retransmission.
		// We do this by adding all the lost bytes to outunsent, and then
		// removing everything already acked.
		s.outunsent.add(start, end)
		for _, a := range s.outacked {
			s.outunsent.sub(a.start, a.end)
		}
	}
}

// dataToSend reports what data should be sent in CRYPTO frames to the peer.
// It calls f with each range of data to send.
// f uses sendData to get the bytes to send, and returns the number of bytes sent.
// dataToSend calls f until no data is left, or f returns 0.
//
// This function is unusually indirect (why not just return a []byte,
// or implement io.Reader?).
//
// Returning a []byte to the caller either requires that we store the
// data to send contiguously (which we don't), allocate a temporary buffer
// and copy into it (inefficient), or return less data than we have available
// (requires complexity to avoid unnecessarily breaking data across frames).
//
// Accepting a []byte from the caller (io.Reader) makes packet construction
// difficult. Since CRYPTO data is encoded with a varint length prefix, the
// location of the data depends on the length of the data. (We could hardcode
// a 2-byte length, of course.)
//
// Instead, we tell the caller how much data is, the caller figures out where
// to put it (and possibly decides that it doesn't have space for this data
// in the packet after all), and the caller then makes a separate call to
// copy the data it wants into position.
func (s *cryptoStream) dataToSend(pto bool, f func(off, size int64) (sent int64)) {
	for {
		off, size := dataToSend(s.out.start, s.out.end, s.outunsent, s.outacked, pto)
		if size == 0 {
			return
		}
		n := f(off, size)
		if n == 0 || pto {
			return
		}
	}
}

// sendData fills b with data to send to the peer, starting at off,
// and marks the data as sent. The caller must have already ascertained
// that there is data to send in this region using dataToSend.
func (s *cryptoStream) sendData(off int64, b []byte) {
	s.out.copy(off, b)
	s.outunsent.sub(off, off+int64(len(b)))
}

// discardKeys is called when the packet protection keys for the stream are dropped.
func (s *cryptoStream) discardKeys() error {
	if s.in.end-s.in.start != 0 {
		// The peer sent some unprocessed CRYPTO data that we're about to discard.
		// Close the connection with a TLS unexpected_message alert.
		// https://www.rfc-editor.org/rfc/rfc5246#section-7.2.2
		const unexpectedMessage = 10
		return localTransportError{
			code:   errTLSBase + unexpectedMessage,
			reason: "excess crypto data",
		}
	}
	// Discard any unacked (but presumably received) data in our output buffer.
	s.out.discardBefore(s.out.end)
	*s = cryptoStream{}
	return nil
}
