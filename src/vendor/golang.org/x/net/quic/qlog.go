// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package quic

import (
	"context"
	"encoding/hex"
	"log/slog"
	"net/netip"
	"time"
)

// Log levels for qlog events.
const (
	// QLogLevelFrame includes per-frame information.
	// When this level is enabled, packet_sent and packet_received events will
	// contain information on individual frames sent/received.
	QLogLevelFrame = slog.Level(-6)

	// QLogLevelPacket events occur at most once per packet sent or received.
	//
	// For example: packet_sent, packet_received.
	QLogLevelPacket = slog.Level(-4)

	// QLogLevelConn events occur multiple times over a connection's lifetime,
	// but less often than the frequency of individual packets.
	//
	// For example: connection_state_updated.
	QLogLevelConn = slog.Level(-2)

	// QLogLevelEndpoint events occur at most once per connection.
	//
	// For example: connection_started, connection_closed.
	QLogLevelEndpoint = slog.Level(0)
)

func (c *Conn) logEnabled(level slog.Level) bool {
	return logEnabled(c.log, level)
}

func logEnabled(log *slog.Logger, level slog.Level) bool {
	return log != nil && log.Enabled(context.Background(), level)
}

// slogHexstring returns a slog.Attr for a value of the hexstring type.
//
// https://www.ietf.org/archive/id/draft-ietf-quic-qlog-main-schema-04.html#section-1.1.1
func slogHexstring(key string, value []byte) slog.Attr {
	return slog.String(key, hex.EncodeToString(value))
}

func slogAddr(key string, value netip.Addr) slog.Attr {
	return slog.String(key, value.String())
}

func (c *Conn) logConnectionStarted(originalDstConnID []byte, peerAddr netip.AddrPort) {
	if c.config.QLogLogger == nil ||
		!c.config.QLogLogger.Enabled(context.Background(), QLogLevelEndpoint) {
		return
	}
	var vantage string
	if c.side == clientSide {
		vantage = "client"
		originalDstConnID = c.connIDState.originalDstConnID
	} else {
		vantage = "server"
	}
	// A qlog Trace container includes some metadata (title, description, vantage_point)
	// and a list of Events. The Trace also includes a common_fields field setting field
	// values common to all events in the trace.
	//
	//	Trace = {
	//	    ? title: text
	//	    ? description: text
	//	    ? configuration: Configuration
	//	    ? common_fields: CommonFields
	//	    ? vantage_point: VantagePoint
	//	    events: [* Event]
	//	}
	//
	// To map this into slog's data model, we start each per-connection trace with a With
	// call that includes both the trace metadata and the common fields.
	//
	// This means that in slog's model, each trace event will also include
	// the Trace metadata fields (vantage_point), which is a divergence from the qlog model.
	c.log = c.config.QLogLogger.With(
		// The group_id permits associating traces taken from different vantage points
		// for the same connection.
		//
		// We use the original destination connection ID as the group ID.
		//
		// https://www.ietf.org/archive/id/draft-ietf-quic-qlog-main-schema-04.html#section-3.4.6
		slogHexstring("group_id", originalDstConnID),
		slog.Group("vantage_point",
			slog.String("name", "go quic"),
			slog.String("type", vantage),
		),
	)
	localAddr := c.endpoint.LocalAddr()
	// https://www.ietf.org/archive/id/draft-ietf-quic-qlog-quic-events-03.html#section-4.2
	c.log.LogAttrs(context.Background(), QLogLevelEndpoint,
		"connectivity:connection_started",
		slogAddr("src_ip", localAddr.Addr()),
		slog.Int("src_port", int(localAddr.Port())),
		slogHexstring("src_cid", c.connIDState.local[0].cid),
		slogAddr("dst_ip", peerAddr.Addr()),
		slog.Int("dst_port", int(peerAddr.Port())),
		slogHexstring("dst_cid", c.connIDState.remote[0].cid),
	)
}

func (c *Conn) logConnectionClosed() {
	if !c.logEnabled(QLogLevelEndpoint) {
		return
	}
	err := c.lifetime.finalErr
	trigger := "error"
	switch e := err.(type) {
	case *ApplicationError:
		// TODO: Distinguish between peer and locally-initiated close.
		trigger = "application"
	case localTransportError:
		switch err {
		case errHandshakeTimeout:
			trigger = "handshake_timeout"
		default:
			if e.code == errNo {
				trigger = "clean"
			}
		}
	case peerTransportError:
		if e.code == errNo {
			trigger = "clean"
		}
	default:
		switch err {
		case errIdleTimeout:
			trigger = "idle_timeout"
		case errStatelessReset:
			trigger = "stateless_reset"
		}
	}
	// https://www.ietf.org/archive/id/draft-ietf-quic-qlog-quic-events-03.html#section-4.3
	c.log.LogAttrs(context.Background(), QLogLevelEndpoint,
		"connectivity:connection_closed",
		slog.String("trigger", trigger),
	)
}

func (c *Conn) logPacketDropped(dgram *datagram) {
	c.log.LogAttrs(context.Background(), QLogLevelPacket,
		"connectivity:packet_dropped",
	)
}

func (c *Conn) logLongPacketReceived(p longPacket, pkt []byte) {
	var frames slog.Attr
	if c.logEnabled(QLogLevelFrame) {
		frames = c.packetFramesAttr(p.payload)
	}
	c.log.LogAttrs(context.Background(), QLogLevelPacket,
		"transport:packet_received",
		slog.Group("header",
			slog.String("packet_type", p.ptype.qlogString()),
			slog.Uint64("packet_number", uint64(p.num)),
			slog.Uint64("flags", uint64(pkt[0])),
			slogHexstring("scid", p.srcConnID),
			slogHexstring("dcid", p.dstConnID),
		),
		slog.Group("raw",
			slog.Int("length", len(pkt)),
		),
		frames,
	)
}

func (c *Conn) log1RTTPacketReceived(p shortPacket, pkt []byte) {
	var frames slog.Attr
	if c.logEnabled(QLogLevelFrame) {
		frames = c.packetFramesAttr(p.payload)
	}
	dstConnID, _ := dstConnIDForDatagram(pkt)
	c.log.LogAttrs(context.Background(), QLogLevelPacket,
		"transport:packet_received",
		slog.Group("header",
			slog.String("packet_type", packetType1RTT.qlogString()),
			slog.Uint64("packet_number", uint64(p.num)),
			slog.Uint64("flags", uint64(pkt[0])),
			slogHexstring("dcid", dstConnID),
		),
		slog.Group("raw",
			slog.Int("length", len(pkt)),
		),
		frames,
	)
}

func (c *Conn) logPacketSent(ptype packetType, pnum packetNumber, src, dst []byte, pktLen int, payload []byte) {
	var frames slog.Attr
	if c.logEnabled(QLogLevelFrame) {
		frames = c.packetFramesAttr(payload)
	}
	var scid slog.Attr
	if len(src) > 0 {
		scid = slogHexstring("scid", src)
	}
	c.log.LogAttrs(context.Background(), QLogLevelPacket,
		"transport:packet_sent",
		slog.Group("header",
			slog.String("packet_type", ptype.qlogString()),
			slog.Uint64("packet_number", uint64(pnum)),
			scid,
			slogHexstring("dcid", dst),
		),
		slog.Group("raw",
			slog.Int("length", pktLen),
		),
		frames,
	)
}

// packetFramesAttr returns the "frames" attribute containing the frames in a packet.
// We currently pass this as a slog Any containing a []slog.Value,
// where each Value is a debugFrame that implements slog.LogValuer.
//
// This isn't tremendously efficient, but avoids the need to put a JSON encoder
// in the quic package or a frame parser in the qlog package.
func (c *Conn) packetFramesAttr(payload []byte) slog.Attr {
	var frames []slog.Value
	for len(payload) > 0 {
		f, n := parseDebugFrame(payload)
		if n < 0 {
			break
		}
		payload = payload[n:]
		switch f := f.(type) {
		case debugFrameAck:
			// The qlog ACK frame contains the ACK Delay field as a duration.
			// Interpreting the contents of this field as a duration requires
			// knowing the peer's ack_delay_exponent transport parameter,
			// and it's possible for us to parse an ACK frame before we've
			// received that parameter.
			//
			// We could plumb connection state down into the frame parser,
			// but for now let's minimize the amount of code that needs to
			// deal with this and convert the unscaled value into a scaled one here.
			ackDelay := time.Duration(-1)
			if c.peerAckDelayExponent >= 0 {
				ackDelay = f.ackDelay.Duration(uint8(c.peerAckDelayExponent))
			}
			frames = append(frames, slog.AnyValue(debugFrameScaledAck{
				ranges:   f.ranges,
				ackDelay: ackDelay,
			}))
		default:
			frames = append(frames, slog.AnyValue(f))
		}
	}
	return slog.Any("frames", frames)
}

func (c *Conn) logPacketLost(space numberSpace, sent *sentPacket) {
	c.log.LogAttrs(context.Background(), QLogLevelPacket,
		"recovery:packet_lost",
		slog.Group("header",
			slog.String("packet_type", sent.ptype.qlogString()),
			slog.Uint64("packet_number", uint64(sent.num)),
		),
	)
}
