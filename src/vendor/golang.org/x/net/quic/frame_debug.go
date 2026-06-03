// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package quic

import (
	"fmt"
	"log/slog"
	"strconv"
	"time"
)

// A debugFrame is a representation of the contents of a QUIC frame,
// used for debug logs and testing but not the primary serving path.
type debugFrame interface {
	String() string
	write(w *packetWriter) bool
	LogValue() slog.Value
}

func parseDebugFrame(b []byte) (f debugFrame, n int) {
	if len(b) == 0 {
		return nil, -1
	}
	switch b[0] {
	case frameTypePadding:
		f, n = parseDebugFramePadding(b)
	case frameTypePing:
		f, n = parseDebugFramePing(b)
	case frameTypeAck, frameTypeAckECN:
		f, n = parseDebugFrameAck(b)
	case frameTypeResetStream:
		f, n = parseDebugFrameResetStream(b)
	case frameTypeStopSending:
		f, n = parseDebugFrameStopSending(b)
	case frameTypeCrypto:
		f, n = parseDebugFrameCrypto(b)
	case frameTypeNewToken:
		f, n = parseDebugFrameNewToken(b)
	case frameTypeStreamBase, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f:
		f, n = parseDebugFrameStream(b)
	case frameTypeMaxData:
		f, n = parseDebugFrameMaxData(b)
	case frameTypeMaxStreamData:
		f, n = parseDebugFrameMaxStreamData(b)
	case frameTypeMaxStreamsBidi, frameTypeMaxStreamsUni:
		f, n = parseDebugFrameMaxStreams(b)
	case frameTypeDataBlocked:
		f, n = parseDebugFrameDataBlocked(b)
	case frameTypeStreamDataBlocked:
		f, n = parseDebugFrameStreamDataBlocked(b)
	case frameTypeStreamsBlockedBidi, frameTypeStreamsBlockedUni:
		f, n = parseDebugFrameStreamsBlocked(b)
	case frameTypeNewConnectionID:
		f, n = parseDebugFrameNewConnectionID(b)
	case frameTypeRetireConnectionID:
		f, n = parseDebugFrameRetireConnectionID(b)
	case frameTypePathChallenge:
		f, n = parseDebugFramePathChallenge(b)
	case frameTypePathResponse:
		f, n = parseDebugFramePathResponse(b)
	case frameTypeConnectionCloseTransport:
		f, n = parseDebugFrameConnectionCloseTransport(b)
	case frameTypeConnectionCloseApplication:
		f, n = parseDebugFrameConnectionCloseApplication(b)
	case frameTypeHandshakeDone:
		f, n = parseDebugFrameHandshakeDone(b)
	default:
		return nil, -1
	}
	return f, n
}

// debugFramePadding is a sequence of PADDING frames.
type debugFramePadding struct {
	size int
	to   int // alternate for writing packets: pad to
}

func parseDebugFramePadding(b []byte) (f debugFramePadding, n int) {
	for n < len(b) && b[n] == frameTypePadding {
		n++
	}
	f.size = n
	return f, n
}

func (f debugFramePadding) String() string {
	return fmt.Sprintf("PADDING*%v", f.size)
}

func (f debugFramePadding) write(w *packetWriter) bool {
	if w.avail() == 0 {
		return false
	}
	if f.to > 0 {
		w.appendPaddingTo(f.to)
		return true
	}
	for i := 0; i < f.size && w.avail() > 0; i++ {
		w.b = append(w.b, frameTypePadding)
	}
	return true
}

func (f debugFramePadding) LogValue() slog.Value {
	return slog.GroupValue(
		slog.String("frame_type", "padding"),
		slog.Int("length", f.size),
	)
}

// debugFramePing is a PING frame.
type debugFramePing struct{}

func parseDebugFramePing(b []byte) (f debugFramePing, n int) {
	return f, 1
}

func (f debugFramePing) String() string {
	return "PING"
}

func (f debugFramePing) write(w *packetWriter) bool {
	return w.appendPingFrame()
}

func (f debugFramePing) LogValue() slog.Value {
	return slog.GroupValue(
		slog.String("frame_type", "ping"),
	)
}

// debugFrameAck is an ACK frame.
type debugFrameAck struct {
	ackDelay unscaledAckDelay
	ranges   []i64range[packetNumber]
	ecn      ecnCounts
}

func parseDebugFrameAck(b []byte) (f debugFrameAck, n int) {
	f.ranges = nil
	_, f.ackDelay, f.ecn, n = consumeAckFrame(b, func(_ int, start, end packetNumber) {
		f.ranges = append(f.ranges, i64range[packetNumber]{
			start: start,
			end:   end,
		})
	})
	// Ranges are parsed high to low; reverse ranges slice to order them low to high.
	for i := 0; i < len(f.ranges)/2; i++ {
		j := len(f.ranges) - 1
		f.ranges[i], f.ranges[j] = f.ranges[j], f.ranges[i]
	}
	return f, n
}

func (f debugFrameAck) String() string {
	s := fmt.Sprintf("ACK Delay=%v", f.ackDelay)
	for _, r := range f.ranges {
		s += fmt.Sprintf(" [%v,%v)", r.start, r.end)
	}

	if (f.ecn != ecnCounts{}) {
		s += fmt.Sprintf(" ECN=[%d,%d,%d]", f.ecn.t0, f.ecn.t1, f.ecn.ce)
	}
	return s
}

func (f debugFrameAck) write(w *packetWriter) bool {
	return w.appendAckFrame(rangeset[packetNumber](f.ranges), f.ackDelay, f.ecn)
}

func (f debugFrameAck) LogValue() slog.Value {
	return slog.StringValue("error: debugFrameAck should not appear as a slog Value")
}

// debugFrameScaledAck is an ACK frame with scaled ACK Delay.
//
// This type is used in qlog events, which need access to the delay as a duration.
type debugFrameScaledAck struct {
	ackDelay time.Duration
	ranges   []i64range[packetNumber]
}

func (f debugFrameScaledAck) LogValue() slog.Value {
	var ackDelay slog.Attr
	if f.ackDelay >= 0 {
		ackDelay = slog.Duration("ack_delay", f.ackDelay)
	}
	return slog.GroupValue(
		slog.String("frame_type", "ack"),
		// Rather than trying to convert the ack ranges into the slog data model,
		// pass a value that can JSON-encode itself.
		slog.Any("acked_ranges", debugAckRanges(f.ranges)),
		ackDelay,
	)
}

type debugAckRanges []i64range[packetNumber]

// AppendJSON appends a JSON encoding of the ack ranges to b, and returns it.
// This is different than the standard json.Marshaler, but more efficient.
// Since we only use this in cooperation with the qlog package,
// encoding/json compatibility is irrelevant.
func (r debugAckRanges) AppendJSON(b []byte) []byte {
	b = append(b, '[')
	for i, ar := range r {
		start, end := ar.start, ar.end-1 // qlog ranges are closed-closed
		if i != 0 {
			b = append(b, ',')
		}
		b = append(b, '[')
		b = strconv.AppendInt(b, int64(start), 10)
		if start != end {
			b = append(b, ',')
			b = strconv.AppendInt(b, int64(end), 10)
		}
		b = append(b, ']')
	}
	b = append(b, ']')
	return b
}

func (r debugAckRanges) String() string {
	return string(r.AppendJSON(nil))
}

// debugFrameResetStream is a RESET_STREAM frame.
type debugFrameResetStream struct {
	id        streamID
	code      uint64
	finalSize int64
}

func parseDebugFrameResetStream(b []byte) (f debugFrameResetStream, n int) {
	f.id, f.code, f.finalSize, n = consumeResetStreamFrame(b)
	return f, n
}

func (f debugFrameResetStream) String() string {
	return fmt.Sprintf("RESET_STREAM ID=%v Code=%v FinalSize=%v", f.id, f.code, f.finalSize)
}

func (f debugFrameResetStream) write(w *packetWriter) bool {
	return w.appendResetStreamFrame(f.id, f.code, f.finalSize)
}

func (f debugFrameResetStream) LogValue() slog.Value {
	return slog.GroupValue(
		slog.String("frame_type", "reset_stream"),
		slog.Uint64("stream_id", uint64(f.id)),
		slog.Uint64("final_size", uint64(f.finalSize)),
	)
}

// debugFrameStopSending is a STOP_SENDING frame.
type debugFrameStopSending struct {
	id   streamID
	code uint64
}

func parseDebugFrameStopSending(b []byte) (f debugFrameStopSending, n int) {
	f.id, f.code, n = consumeStopSendingFrame(b)
	return f, n
}

func (f debugFrameStopSending) String() string {
	return fmt.Sprintf("STOP_SENDING ID=%v Code=%v", f.id, f.code)
}

func (f debugFrameStopSending) write(w *packetWriter) bool {
	return w.appendStopSendingFrame(f.id, f.code)
}

func (f debugFrameStopSending) LogValue() slog.Value {
	return slog.GroupValue(
		slog.String("frame_type", "stop_sending"),
		slog.Uint64("stream_id", uint64(f.id)),
		slog.Uint64("error_code", uint64(f.code)),
	)
}

// debugFrameCrypto is a CRYPTO frame.
type debugFrameCrypto struct {
	off  int64
	data []byte
}

func parseDebugFrameCrypto(b []byte) (f debugFrameCrypto, n int) {
	f.off, f.data, n = consumeCryptoFrame(b)
	return f, n
}

func (f debugFrameCrypto) String() string {
	return fmt.Sprintf("CRYPTO Offset=%v Length=%v", f.off, len(f.data))
}

func (f debugFrameCrypto) write(w *packetWriter) bool {
	b, added := w.appendCryptoFrame(f.off, len(f.data))
	copy(b, f.data)
	return added
}

func (f debugFrameCrypto) LogValue() slog.Value {
	return slog.GroupValue(
		slog.String("frame_type", "crypto"),
		slog.Int64("offset", f.off),
		slog.Int("length", len(f.data)),
	)
}

// debugFrameNewToken is a NEW_TOKEN frame.
type debugFrameNewToken struct {
	token []byte
}

func parseDebugFrameNewToken(b []byte) (f debugFrameNewToken, n int) {
	f.token, n = consumeNewTokenFrame(b)
	return f, n
}

func (f debugFrameNewToken) String() string {
	return fmt.Sprintf("NEW_TOKEN Token=%x", f.token)
}

func (f debugFrameNewToken) write(w *packetWriter) bool {
	return w.appendNewTokenFrame(f.token)
}

func (f debugFrameNewToken) LogValue() slog.Value {
	return slog.GroupValue(
		slog.String("frame_type", "new_token"),
		slogHexstring("token", f.token),
	)
}

// debugFrameStream is a STREAM frame.
type debugFrameStream struct {
	id   streamID
	fin  bool
	off  int64
	data []byte
}

func parseDebugFrameStream(b []byte) (f debugFrameStream, n int) {
	f.id, f.off, f.fin, f.data, n = consumeStreamFrame(b)
	return f, n
}

func (f debugFrameStream) String() string {
	fin := ""
	if f.fin {
		fin = " FIN"
	}
	return fmt.Sprintf("STREAM ID=%v%v Offset=%v Length=%v", f.id, fin, f.off, len(f.data))
}

func (f debugFrameStream) write(w *packetWriter) bool {
	b, added := w.appendStreamFrame(f.id, f.off, len(f.data), f.fin)
	copy(b, f.data)
	return added
}

func (f debugFrameStream) LogValue() slog.Value {
	var fin slog.Attr
	if f.fin {
		fin = slog.Bool("fin", true)
	}
	return slog.GroupValue(
		slog.String("frame_type", "stream"),
		slog.Uint64("stream_id", uint64(f.id)),
		slog.Int64("offset", f.off),
		slog.Int("length", len(f.data)),
		fin,
	)
}

// debugFrameMaxData is a MAX_DATA frame.
type debugFrameMaxData struct {
	max int64
}

func parseDebugFrameMaxData(b []byte) (f debugFrameMaxData, n int) {
	f.max, n = consumeMaxDataFrame(b)
	return f, n
}

func (f debugFrameMaxData) String() string {
	return fmt.Sprintf("MAX_DATA Max=%v", f.max)
}

func (f debugFrameMaxData) write(w *packetWriter) bool {
	return w.appendMaxDataFrame(f.max)
}

func (f debugFrameMaxData) LogValue() slog.Value {
	return slog.GroupValue(
		slog.String("frame_type", "max_data"),
		slog.Int64("maximum", f.max),
	)
}

// debugFrameMaxStreamData is a MAX_STREAM_DATA frame.
type debugFrameMaxStreamData struct {
	id  streamID
	max int64
}

func parseDebugFrameMaxStreamData(b []byte) (f debugFrameMaxStreamData, n int) {
	f.id, f.max, n = consumeMaxStreamDataFrame(b)
	return f, n
}

func (f debugFrameMaxStreamData) String() string {
	return fmt.Sprintf("MAX_STREAM_DATA ID=%v Max=%v", f.id, f.max)
}

func (f debugFrameMaxStreamData) write(w *packetWriter) bool {
	return w.appendMaxStreamDataFrame(f.id, f.max)
}

func (f debugFrameMaxStreamData) LogValue() slog.Value {
	return slog.GroupValue(
		slog.String("frame_type", "max_stream_data"),
		slog.Uint64("stream_id", uint64(f.id)),
		slog.Int64("maximum", f.max),
	)
}

// debugFrameMaxStreams is a MAX_STREAMS frame.
type debugFrameMaxStreams struct {
	streamType streamType
	max        int64
}

func parseDebugFrameMaxStreams(b []byte) (f debugFrameMaxStreams, n int) {
	f.streamType, f.max, n = consumeMaxStreamsFrame(b)
	return f, n
}

func (f debugFrameMaxStreams) String() string {
	return fmt.Sprintf("MAX_STREAMS Type=%v Max=%v", f.streamType, f.max)
}

func (f debugFrameMaxStreams) write(w *packetWriter) bool {
	return w.appendMaxStreamsFrame(f.streamType, f.max)
}

func (f debugFrameMaxStreams) LogValue() slog.Value {
	return slog.GroupValue(
		slog.String("frame_type", "max_streams"),
		slog.String("stream_type", f.streamType.qlogString()),
		slog.Int64("maximum", f.max),
	)
}

// debugFrameDataBlocked is a DATA_BLOCKED frame.
type debugFrameDataBlocked struct {
	max int64
}

func parseDebugFrameDataBlocked(b []byte) (f debugFrameDataBlocked, n int) {
	f.max, n = consumeDataBlockedFrame(b)
	return f, n
}

func (f debugFrameDataBlocked) String() string {
	return fmt.Sprintf("DATA_BLOCKED Max=%v", f.max)
}

func (f debugFrameDataBlocked) write(w *packetWriter) bool {
	return w.appendDataBlockedFrame(f.max)
}

func (f debugFrameDataBlocked) LogValue() slog.Value {
	return slog.GroupValue(
		slog.String("frame_type", "data_blocked"),
		slog.Int64("limit", f.max),
	)
}

// debugFrameStreamDataBlocked is a STREAM_DATA_BLOCKED frame.
type debugFrameStreamDataBlocked struct {
	id  streamID
	max int64
}

func parseDebugFrameStreamDataBlocked(b []byte) (f debugFrameStreamDataBlocked, n int) {
	f.id, f.max, n = consumeStreamDataBlockedFrame(b)
	return f, n
}

func (f debugFrameStreamDataBlocked) String() string {
	return fmt.Sprintf("STREAM_DATA_BLOCKED ID=%v Max=%v", f.id, f.max)
}

func (f debugFrameStreamDataBlocked) write(w *packetWriter) bool {
	return w.appendStreamDataBlockedFrame(f.id, f.max)
}

func (f debugFrameStreamDataBlocked) LogValue() slog.Value {
	return slog.GroupValue(
		slog.String("frame_type", "stream_data_blocked"),
		slog.Uint64("stream_id", uint64(f.id)),
		slog.Int64("limit", f.max),
	)
}

// debugFrameStreamsBlocked is a STREAMS_BLOCKED frame.
type debugFrameStreamsBlocked struct {
	streamType streamType
	max        int64
}

func parseDebugFrameStreamsBlocked(b []byte) (f debugFrameStreamsBlocked, n int) {
	f.streamType, f.max, n = consumeStreamsBlockedFrame(b)
	return f, n
}

func (f debugFrameStreamsBlocked) String() string {
	return fmt.Sprintf("STREAMS_BLOCKED Type=%v Max=%v", f.streamType, f.max)
}

func (f debugFrameStreamsBlocked) write(w *packetWriter) bool {
	return w.appendStreamsBlockedFrame(f.streamType, f.max)
}

func (f debugFrameStreamsBlocked) LogValue() slog.Value {
	return slog.GroupValue(
		slog.String("frame_type", "streams_blocked"),
		slog.String("stream_type", f.streamType.qlogString()),
		slog.Int64("limit", f.max),
	)
}

// debugFrameNewConnectionID is a NEW_CONNECTION_ID frame.
type debugFrameNewConnectionID struct {
	seq           int64
	retirePriorTo int64
	connID        []byte
	token         statelessResetToken
}

func parseDebugFrameNewConnectionID(b []byte) (f debugFrameNewConnectionID, n int) {
	f.seq, f.retirePriorTo, f.connID, f.token, n = consumeNewConnectionIDFrame(b)
	return f, n
}

func (f debugFrameNewConnectionID) String() string {
	return fmt.Sprintf("NEW_CONNECTION_ID Seq=%v Retire=%v ID=%x Token=%x", f.seq, f.retirePriorTo, f.connID, f.token[:])
}

func (f debugFrameNewConnectionID) write(w *packetWriter) bool {
	return w.appendNewConnectionIDFrame(f.seq, f.retirePriorTo, f.connID, f.token)
}

func (f debugFrameNewConnectionID) LogValue() slog.Value {
	return slog.GroupValue(
		slog.String("frame_type", "new_connection_id"),
		slog.Int64("sequence_number", f.seq),
		slog.Int64("retire_prior_to", f.retirePriorTo),
		slogHexstring("connection_id", f.connID),
		slogHexstring("stateless_reset_token", f.token[:]),
	)
}

// debugFrameRetireConnectionID is a NEW_CONNECTION_ID frame.
type debugFrameRetireConnectionID struct {
	seq int64
}

func parseDebugFrameRetireConnectionID(b []byte) (f debugFrameRetireConnectionID, n int) {
	f.seq, n = consumeRetireConnectionIDFrame(b)
	return f, n
}

func (f debugFrameRetireConnectionID) String() string {
	return fmt.Sprintf("RETIRE_CONNECTION_ID Seq=%v", f.seq)
}

func (f debugFrameRetireConnectionID) write(w *packetWriter) bool {
	return w.appendRetireConnectionIDFrame(f.seq)
}

func (f debugFrameRetireConnectionID) LogValue() slog.Value {
	return slog.GroupValue(
		slog.String("frame_type", "retire_connection_id"),
		slog.Int64("sequence_number", f.seq),
	)
}

// debugFramePathChallenge is a PATH_CHALLENGE frame.
type debugFramePathChallenge struct {
	data pathChallengeData
}

func parseDebugFramePathChallenge(b []byte) (f debugFramePathChallenge, n int) {
	f.data, n = consumePathChallengeFrame(b)
	return f, n
}

func (f debugFramePathChallenge) String() string {
	return fmt.Sprintf("PATH_CHALLENGE Data=%x", f.data)
}

func (f debugFramePathChallenge) write(w *packetWriter) bool {
	return w.appendPathChallengeFrame(f.data)
}

func (f debugFramePathChallenge) LogValue() slog.Value {
	return slog.GroupValue(
		slog.String("frame_type", "path_challenge"),
		slog.String("data", fmt.Sprintf("%x", f.data)),
	)
}

// debugFramePathResponse is a PATH_RESPONSE frame.
type debugFramePathResponse struct {
	data pathChallengeData
}

func parseDebugFramePathResponse(b []byte) (f debugFramePathResponse, n int) {
	f.data, n = consumePathResponseFrame(b)
	return f, n
}

func (f debugFramePathResponse) String() string {
	return fmt.Sprintf("PATH_RESPONSE Data=%x", f.data)
}

func (f debugFramePathResponse) write(w *packetWriter) bool {
	return w.appendPathResponseFrame(f.data)
}

func (f debugFramePathResponse) LogValue() slog.Value {
	return slog.GroupValue(
		slog.String("frame_type", "path_response"),
		slog.String("data", fmt.Sprintf("%x", f.data)),
	)
}

// debugFrameConnectionCloseTransport is a CONNECTION_CLOSE frame carrying a transport error.
type debugFrameConnectionCloseTransport struct {
	code      transportError
	frameType uint64
	reason    string
}

func parseDebugFrameConnectionCloseTransport(b []byte) (f debugFrameConnectionCloseTransport, n int) {
	f.code, f.frameType, f.reason, n = consumeConnectionCloseTransportFrame(b)
	return f, n
}

func (f debugFrameConnectionCloseTransport) String() string {
	s := fmt.Sprintf("CONNECTION_CLOSE Code=%v", f.code)
	if f.frameType != 0 {
		s += fmt.Sprintf(" FrameType=%v", f.frameType)
	}
	if f.reason != "" {
		s += fmt.Sprintf(" Reason=%q", f.reason)
	}
	return s
}

func (f debugFrameConnectionCloseTransport) write(w *packetWriter) bool {
	return w.appendConnectionCloseTransportFrame(f.code, f.frameType, f.reason)
}

func (f debugFrameConnectionCloseTransport) LogValue() slog.Value {
	return slog.GroupValue(
		slog.String("frame_type", "connection_close"),
		slog.String("error_space", "transport"),
		slog.Uint64("error_code_value", uint64(f.code)),
		slog.String("reason", f.reason),
	)
}

// debugFrameConnectionCloseApplication is a CONNECTION_CLOSE frame carrying an application error.
type debugFrameConnectionCloseApplication struct {
	code   uint64
	reason string
}

func parseDebugFrameConnectionCloseApplication(b []byte) (f debugFrameConnectionCloseApplication, n int) {
	f.code, f.reason, n = consumeConnectionCloseApplicationFrame(b)
	return f, n
}

func (f debugFrameConnectionCloseApplication) String() string {
	s := fmt.Sprintf("CONNECTION_CLOSE AppCode=%v", f.code)
	if f.reason != "" {
		s += fmt.Sprintf(" Reason=%q", f.reason)
	}
	return s
}

func (f debugFrameConnectionCloseApplication) write(w *packetWriter) bool {
	return w.appendConnectionCloseApplicationFrame(f.code, f.reason)
}

func (f debugFrameConnectionCloseApplication) LogValue() slog.Value {
	return slog.GroupValue(
		slog.String("frame_type", "connection_close"),
		slog.String("error_space", "application"),
		slog.Uint64("error_code_value", uint64(f.code)),
		slog.String("reason", f.reason),
	)
}

// debugFrameHandshakeDone is a HANDSHAKE_DONE frame.
type debugFrameHandshakeDone struct{}

func parseDebugFrameHandshakeDone(b []byte) (f debugFrameHandshakeDone, n int) {
	return f, 1
}

func (f debugFrameHandshakeDone) String() string {
	return "HANDSHAKE_DONE"
}

func (f debugFrameHandshakeDone) write(w *packetWriter) bool {
	return w.appendHandshakeDoneFrame()
}

func (f debugFrameHandshakeDone) LogValue() slog.Value {
	return slog.GroupValue(
		slog.String("frame_type", "handshake_done"),
	)
}
