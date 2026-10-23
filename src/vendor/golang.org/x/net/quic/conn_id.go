// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package quic

import (
	"bytes"
	"crypto/rand"
	"net/netip"
	"slices"
)

// connIDState is a conn's connection IDs.
type connIDState struct {
	// The destination connection IDs of packets we receive are local.
	// The destination connection IDs of packets we send are remote.
	//
	// Local IDs are usually issued by us, and remote IDs by the peer.
	// The exception is the transient destination connection ID sent in
	// a client's Initial packets, which is chosen by the client.
	//
	// These are []connID rather than []*connID to minimize allocations.
	local  []connID
	remote []remoteConnID

	nextLocalSeq          int64
	peerActiveConnIDLimit int64 // peer's active_connection_id_limit

	// Handling of retirement of remote connection IDs.
	// The rangesets track ID sequence numbers.
	//
	// All retired IDs are in remoteRetired.
	//
	// IDs in need of retirement are added to remoteRetiringToSend,
	// moved to remoteRetiringSent once we send a RETIRE_CONNECTION_ID frame,
	// and removed from remoteRetiringSent once retirement completes.
	//
	// (We could be more clever and use just two rangesets, at the cost of being clever.)
	remoteRetired        rangeset[int64] // remote IDs retired
	remoteRetiringToSend rangeset[int64] // remote IDs in need of retirement
	remoteRetiringSent   rangeset[int64] // remote IDs waiting for ack of retirement

	originalDstConnID []byte // expected original_destination_connection_id param
	retrySrcConnID    []byte // expected retry_source_connection_id param

	needSend bool
}

// A connID is a connection ID and associated metadata.
type connID struct {
	// cid is the connection ID itself.
	cid []byte

	// seq is the connection ID's sequence number:
	// https://www.rfc-editor.org/rfc/rfc9000.html#section-5.1.1-1
	//
	// For the transient destination ID in a client's Initial packet, this is -1.
	seq int64

	// send is set when the connection ID's state needs to be sent to the peer.
	//
	// For local IDs, this indicates a new ID that should be sent
	// in a NEW_CONNECTION_ID frame.
	//
	// For remote IDs, this indicates a retired ID that should be sent
	// in a RETIRE_CONNECTION_ID frame.
	send sentVal
}

// A remoteConnID is a connection ID and stateless reset token.
type remoteConnID struct {
	connID
	resetToken statelessResetToken
	path       pathAddrs // path this connection ID may be used on
}

func (s *connIDState) initClient(c *Conn, path pathAddrs) error {
	// Client chooses its initial connection ID, and sends it
	// in the Source Connection ID field of the first Initial packet.
	locid, err := c.newConnID(0)
	if err != nil {
		return err
	}
	s.local = append(s.local, connID{
		seq: 0,
		cid: locid,
	})
	s.nextLocalSeq = 1
	c.endpoint.connsMap.updateConnIDs(func(conns *connsMap) {
		conns.addConnID(c, locid)
	})

	// Client chooses an initial, transient connection ID for the server,
	// and sends it in the Destination Connection ID field of the first Initial packet.
	remid, err := c.newConnID(-1)
	if err != nil {
		return err
	}
	s.remote = append(s.remote, remoteConnID{
		connID: connID{
			seq: -1,
			cid: remid,
		},
		path: path,
	})
	s.originalDstConnID = remid
	return nil
}

func (s *connIDState) initServer(c *Conn, path pathAddrs, cids newServerConnIDs) error {
	dstConnID := cloneBytes(cids.dstConnID)
	// Client-chosen, transient connection ID received in the first Initial packet.
	// The server will not use this as the Source Connection ID of packets it sends,
	// but remembers it because it may receive packets sent to this destination.
	s.local = append(s.local, connID{
		seq: -1,
		cid: dstConnID,
	})

	// Server chooses a connection ID, and sends it in the Source Connection ID of
	// the response to the clent.
	locid, err := c.newConnID(0)
	if err != nil {
		return err
	}
	s.local = append(s.local, connID{
		seq: 0,
		cid: locid,
	})
	s.nextLocalSeq = 1
	c.endpoint.connsMap.updateConnIDs(func(conns *connsMap) {
		conns.addConnID(c, dstConnID)
		conns.addConnID(c, locid)
	})

	// Client chose its own connection ID.
	cidPath := path
	if len(cids.srcConnID) == 0 {
		// A zero-length connection ID is not associated with any particular path.
		cidPath = pathAddrs{}
	}
	s.remote = append(s.remote, remoteConnID{
		connID: connID{
			seq: 0,
			cid: cloneBytes(cids.srcConnID),
		},
		path: cidPath,
	})
	return nil
}

// srcConnID is the Source Connection ID to use in a sent packet.
func (s *connIDState) srcConnID() []byte {
	if s.local[0].seq == -1 && len(s.local) > 1 {
		// Don't use the transient connection ID if another is available.
		return s.local[1].cid
	}
	return s.local[0].cid
}

// dstConnIDForPath is the Destination Connection ID to use in a sent packet.
//
// DCIDs are bound to a path once allocated.
// If the path has no DCID bound to it, a new one will be assigned to it (if available).
func (s *connIDState) dstConnIDForPath(path pathAddrs) (dstConnID []byte, ok bool) {
	if !path.peer.IsValid() {
		// Should be impossible, but guard against trying to use a nonsensical path.
		return nil, false
	}
	allocID := -1
	for i := range len(s.remote) {
		if s.remote[i].path == path {
			// We have an existing DCID for this path.
			return s.remote[i].cid, true
		}
		if len(s.remote[i].cid) == 0 {
			// An endpoint with a zero-length DCID uses it for all packets.
			// https://www.rfc-editor.org/rfc/rfc9000.html#section-5.1.1-8
			return []byte{}, true
		}
		if allocID < 0 && s.remote[i].path == (pathAddrs{}) {
			// This is an unused DCID which we might allocate to this path.
			allocID = i
		}
	}
	if allocID < 0 {
		// We have no DCID for this path, and none to allocate to it.
		return nil, false
	}
	// Allocate a new DCID for the path.
	s.remote[allocID].path = path
	return s.remote[allocID].cid, true
}

// abandonPath retires any remote connection IDs associated with path.
func (s *connIDState) abandonPath(c *Conn, path pathAddrs) {
	if path == (pathAddrs{}) {
		return
	}
	s.retireRemoteIDs(c, func(rcid remoteConnID) bool {
		return rcid.path == path
	})
}

// retireRemoteIDs retires all remote connection IDs for which f returns true.
// It never retires a zero-length connection ID (which will be the peer's only ID).
func (s *connIDState) retireRemoteIDs(c *Conn, f func(rcid remoteConnID) bool) {
	var retiredTokens []statelessResetToken
	s.remote = slices.DeleteFunc(s.remote, func(rcid remoteConnID) bool {
		if len(rcid.cid) == 0 || !f(rcid) {
			return false
		}
		if !s.remoteRetired.contains(rcid.seq) {
			s.remoteRetired.add(rcid.seq, rcid.seq+1)
			s.remoteRetiringToSend.add(rcid.seq, rcid.seq+1)
			s.needSend = true
		}
		retiredTokens = append(retiredTokens, rcid.resetToken)
		return true
	})
	if len(retiredTokens) > 0 {
		c.endpoint.connsMap.updateConnIDs(func(conns *connsMap) {
			for _, resetToken := range retiredTokens {
				conns.retireResetToken(c, resetToken)
			}
		})
	}
}

// isValidStatelessResetToken reports whether the given reset token is
// associated with a non-retired connection ID which we have used for the given path.
func (s *connIDState) isValidStatelessResetToken(resetToken statelessResetToken, peer netip.AddrPort) bool {
	if resetToken == (statelessResetToken{}) {
		return false // ignore all-0 token (don't match an uninitialized token)
	}
	if !peer.IsValid() {
		return false
	}
	return slices.ContainsFunc(s.remote, func(rcid remoteConnID) bool {
		// A zero-length connection ID is used for all paths.
		// A non-zero-length connection ID must have been used for this specific path.
		return (len(rcid.cid) == 0 || rcid.path.peer == peer) &&
			rcid.resetToken == resetToken
	})
}

// setPeerActiveConnIDLimit sets the active_connection_id_limit
// transport parameter received from the peer.
func (s *connIDState) setPeerActiveConnIDLimit(c *Conn, lim int64) error {
	s.peerActiveConnIDLimit = lim
	return s.issueLocalIDs(c)
}

func (s *connIDState) issueLocalIDs(c *Conn) error {
	toIssue := min(int(s.peerActiveConnIDLimit), maxPeerActiveConnIDLimit)
	for i := range s.local {
		if s.local[i].seq != -1 {
			toIssue--
		}
	}
	var newIDs [][]byte
	for toIssue > 0 {
		cid, err := c.newConnID(s.nextLocalSeq)
		if err != nil {
			return err
		}
		newIDs = append(newIDs, cid)
		s.local = append(s.local, connID{
			seq: s.nextLocalSeq,
			cid: cid,
		})
		s.local[len(s.local)-1].send.setUnsent()
		s.nextLocalSeq++
		s.needSend = true
		toIssue--
	}
	c.endpoint.connsMap.updateConnIDs(func(conns *connsMap) {
		for _, cid := range newIDs {
			conns.addConnID(c, cid)
		}
	})
	return nil
}

// validateTransportParameters verifies the original_destination_connection_id and
// initial_source_connection_id transport parameters match the expected values.
func (s *connIDState) validateTransportParameters(c *Conn, isRetry bool, p transportParameters) error {
	// TODO: Consider returning more detailed errors, for debugging.
	// Verify original_destination_connection_id matches
	// the transient remote connection ID we chose (client)
	// or is empty (server).
	if !bytes.Equal(s.originalDstConnID, p.originalDstConnID) {
		return localTransportError{
			code:   errTransportParameter,
			reason: "original_destination_connection_id mismatch",
		}
	}
	s.originalDstConnID = nil // we have no further need for this
	// Verify retry_source_connection_id matches the value from
	// the server's Retry packet (when one was sent), or is empty.
	if !bytes.Equal(p.retrySrcConnID, s.retrySrcConnID) {
		return localTransportError{
			code:   errTransportParameter,
			reason: "retry_source_connection_id mismatch",
		}
	}
	s.retrySrcConnID = nil // we have no further need for this
	// Verify initial_source_connection_id matches the first remote connection ID.
	if len(s.remote) == 0 || s.remote[0].seq != 0 {
		return localTransportError{
			code:   errInternal,
			reason: "remote connection id missing",
		}
	}
	if !bytes.Equal(p.initialSrcConnID, s.remote[0].cid) {
		return localTransportError{
			code:   errTransportParameter,
			reason: "initial_source_connection_id mismatch",
		}
	}
	if len(p.statelessResetToken) > 0 {
		if c.side == serverSide {
			return localTransportError{
				code:   errTransportParameter,
				reason: "client sent stateless_reset_token",
			}
		}
		token := statelessResetToken(p.statelessResetToken)
		s.remote[0].resetToken = token
		c.endpoint.connsMap.updateConnIDs(func(conns *connsMap) {
			conns.addResetToken(c, token)
		})
	}
	return nil
}

// handlePacket updates the connection ID state during the handshake
// (Initial and Handshake packets).
func (s *connIDState) handlePacket(c *Conn, ptype packetType, srcConnID []byte) {
	switch {
	case ptype == packetTypeInitial && c.side == clientSide:
		if len(s.remote) == 1 && s.remote[0].seq == -1 {
			// We're a client connection processing the first Initial packet
			// from the server. Replace the transient remote connection ID
			// with the Source Connection ID from the packet.
			cidPath := c.path
			if len(srcConnID) == 0 {
				// A zero-length connection ID is not associated with any particular path.
				cidPath = pathAddrs{}
			}
			s.remote[0] = remoteConnID{
				connID: connID{
					seq: 0,
					cid: cloneBytes(srcConnID),
				},
				path: cidPath,
			}
		}
	case ptype == packetTypeHandshake && c.side == serverSide:
		if len(s.local) > 0 && s.local[0].seq == -1 {
			// We're a server connection processing the first Handshake packet from
			// the client. Discard the transient, client-chosen connection ID used
			// for Initial packets; the client will never send it again.
			cid := s.local[0].cid
			c.endpoint.connsMap.updateConnIDs(func(conns *connsMap) {
				conns.retireConnID(c, cid)
			})
			s.local = append(s.local[:0], s.local[1:]...)
		}
	}
}

func (s *connIDState) handleRetryPacket(srcConnID []byte) {
	if len(s.remote) != 1 || s.remote[0].seq != -1 {
		panic("BUG: handling retry with non-transient remote conn id")
	}
	s.retrySrcConnID = cloneBytes(srcConnID)
	s.remote[0].cid = s.retrySrcConnID
}

func (s *connIDState) handleNewConnID(c *Conn, seq, retirePriorTo int64, cid []byte, resetToken statelessResetToken) error {
	if len(s.remote) > 0 && len(s.remote[0].cid) == 0 {
		// "An endpoint that is sending packets with a zero-length
		// Destination Connection ID MUST treat receipt of a NEW_CONNECTION_ID
		// frame as a connection error of type PROTOCOL_VIOLATION."
		// https://www.rfc-editor.org/rfc/rfc9000.html#section-19.15-6
		return localTransportError{
			code:   errProtocolViolation,
			reason: "NEW_CONNECTION_ID from peer with zero-length DCID",
		}
	}

	// Check for errors (mismatch, exceeding limits) before we touch the list of
	// connection IDs. If something is wrong with this frame, we'll use our existing
	// state to send the error.
	var (
		have      = false // do we already have this connection ID?
		havePath  = false // will we still have a connection ID for the active path?
		unretired = 0
	)
	for i := range s.remote {
		rcid := &s.remote[i]
		if rcid.seq == seq {
			if !bytes.Equal(rcid.cid, cid) {
				return localTransportError{
					code:   errProtocolViolation,
					reason: "NEW_CONNECTION_ID does not match prior id",
				}
			}
			have = true // yes, we've seen this sequence number
		}
		if rcid.seq >= retirePriorTo {
			unretired++
			if rcid.path == c.path {
				havePath = true
			}
		}
	}
	if !have && s.remoteRetired.contains(seq) {
		// This ID was already retired. We have it, in the retired set.
		have = true
	}
	if !have {
		unretired++
	}
	if unretired > activeConnIDLimit {
		// Retired connection IDs (including newly-retired ones) do not count
		// against the limit.
		// https://www.rfc-editor.org/rfc/rfc9000.html#section-5.1.1-5
		return localTransportError{
			code:   errConnectionIDLimit,
			reason: "active_connection_id_limit exceeded",
		}
	}

	if retirePriorTo > s.remoteRetired.rangeContaining(0).end {
		// Add newly-retired connection IDs to the set we need to send
		// RETIRE_CONNECTION_ID frames for, and remove them from s.remote.
		//
		// (This might cause us to send a RETIRE_CONNECTION_ID for an ID we've
		// never seen. That's fine.)
		for r := range s.remoteRetired.iterNotContains() {
			if r.start >= retirePriorTo {
				break
			}
			s.remoteRetiringToSend.add(r.start, min(r.end, retirePriorTo))
			s.needSend = true
		}
		s.remoteRetired.add(0, retirePriorTo)
		s.retireRemoteIDs(c, func(rcid remoteConnID) bool {
			return rcid.seq < retirePriorTo
		})
	}

	// "An endpoint SHOULD limit the number of connection IDs it has retired locally
	// for which RETIRE_CONNECTION_ID frames have not yet been acknowledged."
	// https://www.rfc-editor.org/rfc/rfc9000#section-5.1.2-6
	//
	// Set a limit of three times the active_connection_id_limit for
	// the total number of remote connection IDs we keep retirement state for.
	//
	// (We check this even if the peer hasn't increased retirePriorTo, because a
	// malicious peer could induce us to probe and abandon paths, ever increasing
	// the number of retired connection IDs without itself ever increasing retirePriorTo.
	// Not the most plausible scenario to exploit, admittedly.)
	if s.remoteRetiringToSend.size()+s.remoteRetiringSent.size() > 3*activeConnIDLimit {
		return localTransportError{
			code:   errConnectionIDLimit,
			reason: "too many unacknowledged retired connection ids",
		}
	}

	if !have {
		// This is a new connection ID that we have not seen before.
		//
		// We could take steps to keep the list of remote connection IDs
		// sorted by sequence number, but there's no particular need
		// so we don't bother.
		s.remote = append(s.remote, remoteConnID{
			connID: connID{
				seq: seq,
				cid: cloneBytes(cid),
			},
			resetToken: resetToken,
		})
		c.endpoint.connsMap.updateConnIDs(func(conns *connsMap) {
			conns.addResetToken(c, resetToken)
		})
	}

	if !havePath {
		// We have no connection ID for the active path
		// (probably because we just retired it).
		//
		// Try to allocate a new one now.
		// If we don't have an unused one to allocate, there's not much we can do.
		_, _ = s.dstConnIDForPath(c.path)
	}

	return nil
}

func (s *connIDState) handleRetireConnID(c *Conn, seq int64) error {
	if seq >= s.nextLocalSeq {
		return localTransportError{
			code:   errProtocolViolation,
			reason: "RETIRE_CONNECTION_ID for unissued sequence number",
		}
	}
	for i := range s.local {
		if s.local[i].seq == seq {
			cid := s.local[i].cid
			c.endpoint.connsMap.updateConnIDs(func(conns *connsMap) {
				conns.retireConnID(c, cid)
			})
			s.local = append(s.local[:i], s.local[i+1:]...)
			break
		}
	}
	s.issueLocalIDs(c)
	return nil
}

func (s *connIDState) ackOrLossNewConnectionID(pnum packetNumber, seq int64, fate packetFate) {
	for i := range s.local {
		if s.local[i].seq != seq {
			continue
		}
		s.local[i].send.ackOrLoss(pnum, fate)
		if fate != packetAcked {
			s.needSend = true
		}
		return
	}
}

func (s *connIDState) ackOrLossRetireConnectionID(pnum packetNumber, seq int64, fate packetFate) {
	s.remoteRetiringSent.sub(seq, seq+1)
	if fate == packetLost {
		// RETIRE_CONNECTION_ID frame was lost, mark for retransmission.
		s.remoteRetiringToSend.add(seq, seq+1)
		s.needSend = true
	} else {
		s.remoteRetiringToSend.sub(seq, seq+1)
	}
}

// appendFrames appends NEW_CONNECTION_ID and RETIRE_CONNECTION_ID frames
// to the current packet.
//
// It returns true if no more frames need appending,
// false if not everything fit in the current packet.
func (s *connIDState) appendFrames(c *Conn, pnum packetNumber, pto bool) bool {
	if !s.needSend && !pto {
		// Fast path: We don't need to send anything.
		return true
	}
	retireBefore := int64(0)
	if s.local[0].seq != -1 {
		retireBefore = s.local[0].seq
	}
	for i := range s.local {
		if !s.local[i].send.shouldSendPTO(pto) {
			continue
		}
		if !c.w.appendNewConnectionIDFrame(
			s.local[i].seq,
			retireBefore,
			s.local[i].cid,
			c.endpoint.resetGen.tokenForConnID(s.local[i].cid),
		) {
			return false
		}
		s.local[i].send.setSent(pnum)
	}
	if pto {
		for _, r := range s.remoteRetiringSent {
			for cid := r.start; cid < r.end; cid++ {
				if !c.w.appendRetireConnectionIDFrame(cid) {
					return false
				}
			}
		}
	}
	for s.remoteRetiringToSend.numRanges() > 0 {
		cid := s.remoteRetiringToSend.min()
		if !c.w.appendRetireConnectionIDFrame(cid) {
			return false
		}
		s.remoteRetiringToSend.sub(cid, cid+1)
		s.remoteRetiringSent.add(cid, cid+1)
	}
	s.needSend = false
	return true
}

func cloneBytes(b []byte) []byte {
	n := make([]byte, len(b))
	copy(n, b)
	return n
}

func (c *Conn) newConnID(seq int64) ([]byte, error) {
	if c.testHooks != nil {
		return c.testHooks.newConnID(seq)
	}
	return newRandomConnID(seq)
}

func newRandomConnID(_ int64) ([]byte, error) {
	// It is not necessary for connection IDs to be cryptographically secure,
	// but it doesn't hurt.
	id := make([]byte, connIDLen)
	if _, err := rand.Read(id); err != nil {
		// TODO: Surface this error as a metric or log event or something.
		// rand.Read really shouldn't ever fail, but if it does, we should
		// have a way to inform the user.
		return nil, err
	}
	return id, nil
}
