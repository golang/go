// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http2

import (
	"bytes"
	"fmt"
	"log"
	"net/http/internal/httpcommon"
	"net/url"

	"golang.org/x/net/http/httpguts"
	"golang.org/x/net/http2/hpack"
)

// writeFramer is implemented by any type that is used to write frames.
type writeFramer interface {
	writeFrame(writeContext) error

	// staysWithinBuffer reports whether this writer promises that
	// it will only write less than or equal to size bytes, and it
	// won't Flush the write context.
	staysWithinBuffer(size int) bool
}

// writeContext is the interface needed by the various frame writer
// types below. All the writeFrame methods below are scheduled via the
// frame writing scheduler (see writeScheduler in writesched.go).
//
// This interface is implemented by *serverConn.
//
// TODO: decide whether to a) use this in the client code (which didn't
// end up using this yet, because it has a simpler design, not
// currently implementing priorities), or b) delete this and
// make the server code a bit more concrete.
type writeContext interface {
	Framer() *Framer
	Flush() error
	CloseConn() error
	// HeaderEncoder returns an HPACK encoder that writes to the
	// returned buffer.
	HeaderEncoder() (*hpack.Encoder, *bytes.Buffer)
}

// writeEndsStream reports whether w writes a frame that will transition
// the stream to a half-closed local state. This returns false for RST_STREAM,
// which closes the entire stream (not just the local half).
func writeEndsStream(w writeFramer) bool {
	switch v := w.(type) {
	case *writeData:
		return v.endStream
	case *writeResHeaders:
		return v.endStream
	case nil:
		// This can only happen if the caller reuses w after it's
		// been intentionally nil'ed out to prevent use. Keep this
		// here to catch future refactoring breaking it.
		panic("writeEndsStream called on nil writeFramer")
	}
	return false
}

type flushFrameWriter struct{}

func (flushFrameWriter) writeFrame(ctx writeContext) error {
	return ctx.Flush()
}

func (flushFrameWriter) staysWithinBuffer(max int) bool { return false }

type writeSettings []Setting

func (s writeSettings) staysWithinBuffer(max int) bool {
	const settingSize = 6 // uint16 + uint32
	return frameHeaderLen+settingSize*len(s) <= max

}

func (s writeSettings) writeFrame(ctx writeContext) error {
	return ctx.Framer().WriteSettings([]Setting(s)...)
}

type writeGoAway struct {
	maxStreamID uint32
	code        ErrCode
}

func (p *writeGoAway) writeFrame(ctx writeContext) error {
	err := ctx.Framer().WriteGoAway(p.maxStreamID, p.code, nil)
	ctx.Flush() // ignore error: we're hanging up on them anyway
	return err
}

func (*writeGoAway) staysWithinBuffer(max int) bool { return false } // flushes

type writeData struct {
	streamID  uint32
	p         []byte
	endStream bool
}

func (w *writeData) String() string {
	return fmt.Sprintf("writeData(stream=%d, p=%d, endStream=%v)", w.streamID, len(w.p), w.endStream)
}

func (w *writeData) writeFrame(ctx writeContext) error {
	return ctx.Framer().WriteData(w.streamID, w.endStream, w.p)
}

func (w *writeData) staysWithinBuffer(max int) bool {
	return frameHeaderLen+len(w.p) <= max
}

// handlerPanicRST is the message sent from handler goroutines when
// the handler panics.
type handlerPanicRST struct {
	StreamID uint32
}

func (hp handlerPanicRST) writeFrame(ctx writeContext) error {
	return ctx.Framer().WriteRSTStream(hp.StreamID, ErrCodeInternal)
}

func (hp handlerPanicRST) staysWithinBuffer(max int) bool { return frameHeaderLen+4 <= max }

func (se StreamError) writeFrame(ctx writeContext) error {
	return ctx.Framer().WriteRSTStream(se.StreamID, se.Code)
}

func (se StreamError) staysWithinBuffer(max int) bool { return frameHeaderLen+4 <= max }

type writePing struct {
	data [8]byte
}

func (w writePing) writeFrame(ctx writeContext) error {
	return ctx.Framer().WritePing(false, w.data)
}

func (w writePing) staysWithinBuffer(max int) bool { return frameHeaderLen+len(w.data) <= max }

type writePingAck struct{ pf *PingFrame }

func (w writePingAck) writeFrame(ctx writeContext) error {
	return ctx.Framer().WritePing(true, w.pf.Data)
}

func (w writePingAck) staysWithinBuffer(max int) bool { return frameHeaderLen+len(w.pf.Data) <= max }

type writeSettingsAck struct{}

func (writeSettingsAck) writeFrame(ctx writeContext) error {
	return ctx.Framer().WriteSettingsAck()
}

func (writeSettingsAck) staysWithinBuffer(max int) bool { return frameHeaderLen <= max }

// splitHeaderBlock splits headerBlock into fragments so that each fragment fits
// in a single frame, then calls fn for each fragment. firstFrag/lastFrag are true
// for the first/last fragment, respectively.
func splitHeaderBlock(ctx writeContext, headerBlock []byte, fn func(ctx writeContext, frag []byte, firstFrag, lastFrag bool) error) error {
	// For now we're lazy and just pick the minimum MAX_FRAME_SIZE
	// that all peers must support (16KB). Later we could care
	// more and send larger frames if the peer advertised it, but
	// there's little point. Most headers are small anyway (so we
	// generally won't have CONTINUATION frames), and extra frames
	// only waste 9 bytes anyway.
	const maxFrameSize = 16384

	first := true
	for len(headerBlock) > 0 {
		frag := headerBlock
		if len(frag) > maxFrameSize {
			frag = frag[:maxFrameSize]
		}
		headerBlock = headerBlock[len(frag):]
		if err := fn(ctx, frag, first, len(headerBlock) == 0); err != nil {
			return err
		}
		first = false
	}
	return nil
}

// writeResHeaders is a request to write a HEADERS and 0+ CONTINUATION frames
// for HTTP response headers or trailers from a server handler.
type writeResHeaders struct {
	streamID    uint32
	httpResCode int      // 0 means no ":status" line
	h           Header   // may be nil
	trailers    []string // if non-nil, which keys of h to write. nil means all.
	endStream   bool

	date          string
	contentType   string
	contentLength string
}

func encKV(enc *hpack.Encoder, k, v string) {
	if VerboseLogs {
		log.Printf("http2: server encoding header %q = %q", k, v)
	}
	enc.WriteField(hpack.HeaderField{Name: k, Value: v})
}

func (w *writeResHeaders) staysWithinBuffer(max int) bool {
	// TODO: this is a common one. It'd be nice to return true
	// here and get into the fast path if we could be clever and
	// calculate the size fast enough, or at least a conservative
	// upper bound that usually fires. (Maybe if w.h and
	// w.trailers are nil, so we don't need to enumerate it.)
	// Otherwise I'm afraid that just calculating the length to
	// answer this question would be slower than the ~2µs benefit.
	return false
}

func (w *writeResHeaders) writeFrame(ctx writeContext) error {
	enc, buf := ctx.HeaderEncoder()
	buf.Reset()

	if w.httpResCode != 0 {
		encKV(enc, ":status", httpCodeString(w.httpResCode))
	}

	encodeHeaders(enc, w.h, w.trailers)

	if w.contentType != "" {
		encKV(enc, "content-type", w.contentType)
	}
	if w.contentLength != "" {
		encKV(enc, "content-length", w.contentLength)
	}
	if w.date != "" {
		encKV(enc, "date", w.date)
	}

	headerBlock := buf.Bytes()
	if len(headerBlock) == 0 && w.trailers == nil {
		panic("unexpected empty hpack")
	}

	return splitHeaderBlock(ctx, headerBlock, w.writeHeaderBlock)
}

func (w *writeResHeaders) writeHeaderBlock(ctx writeContext, frag []byte, firstFrag, lastFrag bool) error {
	if firstFrag {
		return ctx.Framer().WriteHeaders(HeadersFrameParam{
			StreamID:      w.streamID,
			BlockFragment: frag,
			EndStream:     w.endStream,
			EndHeaders:    lastFrag,
		})
	} else {
		return ctx.Framer().WriteContinuation(w.streamID, lastFrag, frag)
	}
}

// writePushPromise is a request to write a PUSH_PROMISE and 0+ CONTINUATION frames.
type writePushPromise struct {
	streamID uint32   // pusher stream
	method   string   // for :method
	url      *url.URL // for :scheme, :authority, :path
	h        Header

	// Creates an ID for a pushed stream. This runs on serveG just before
	// the frame is written. The returned ID is copied to promisedID.
	allocatePromisedID func() (uint32, error)
	promisedID         uint32
}

func (w *writePushPromise) staysWithinBuffer(max int) bool {
	// TODO: see writeResHeaders.staysWithinBuffer
	return false
}

func (w *writePushPromise) writeFrame(ctx writeContext) error {
	enc, buf := ctx.HeaderEncoder()
	buf.Reset()

	encKV(enc, ":method", w.method)
	encKV(enc, ":scheme", w.url.Scheme)
	encKV(enc, ":authority", w.url.Host)
	encKV(enc, ":path", w.url.RequestURI())
	encodeHeaders(enc, w.h, nil)

	headerBlock := buf.Bytes()
	if len(headerBlock) == 0 {
		panic("unexpected empty hpack")
	}

	return splitHeaderBlock(ctx, headerBlock, w.writeHeaderBlock)
}

func (w *writePushPromise) writeHeaderBlock(ctx writeContext, frag []byte, firstFrag, lastFrag bool) error {
	if firstFrag {
		return ctx.Framer().WritePushPromise(PushPromiseParam{
			StreamID:      w.streamID,
			PromiseID:     w.promisedID,
			BlockFragment: frag,
			EndHeaders:    lastFrag,
		})
	} else {
		return ctx.Framer().WriteContinuation(w.streamID, lastFrag, frag)
	}
}

type write100ContinueHeadersFrame struct {
	streamID uint32
}

func (w write100ContinueHeadersFrame) writeFrame(ctx writeContext) error {
	enc, buf := ctx.HeaderEncoder()
	buf.Reset()
	encKV(enc, ":status", "100")
	return ctx.Framer().WriteHeaders(HeadersFrameParam{
		StreamID:      w.streamID,
		BlockFragment: buf.Bytes(),
		EndStream:     false,
		EndHeaders:    true,
	})
}

func (w write100ContinueHeadersFrame) staysWithinBuffer(max int) bool {
	// Sloppy but conservative:
	return 9+2*(len(":status")+len("100")) <= max
}

type writeWindowUpdate struct {
	streamID uint32 // or 0 for conn-level
	n        uint32
}

func (wu writeWindowUpdate) staysWithinBuffer(max int) bool { return frameHeaderLen+4 <= max }

func (wu writeWindowUpdate) writeFrame(ctx writeContext) error {
	return ctx.Framer().WriteWindowUpdate(wu.streamID, wu.n)
}

// encodeHeaders encodes an http.Header. If keys is not nil, then (k, h[k])
// is encoded only if k is in keys.
func encodeHeaders(enc *hpack.Encoder, h Header, keys []string) {
	if keys == nil {
		sorter := sorterPool.Get().(*sorter)
		// Using defer here, since the returned keys from the
		// sorter.Keys method is only valid until the sorter
		// is returned:
		defer sorterPool.Put(sorter)
		keys = sorter.Keys(h)
	}
	for _, k := range keys {
		vv := h[k]
		k, ascii := httpcommon.LowerHeader(k)
		if !ascii {
			// Skip writing invalid headers. Per RFC 7540, Section 8.1.2, header
			// field names have to be ASCII characters (just as in HTTP/1.x).
			continue
		}
		if !validWireHeaderFieldName(k) {
			// Skip it as backup paranoia. Per
			// golang.org/issue/14048, these should
			// already be rejected at a higher level.
			continue
		}
		isTE := k == "transfer-encoding"
		for _, v := range vv {
			if !httpguts.ValidHeaderFieldValue(v) {
				// TODO: return an error? golang.org/issue/14048
				// For now just omit it.
				continue
			}
			// TODO: more of "8.1.2.2 Connection-Specific Header Fields"
			if isTE && v != "trailers" {
				continue
			}
			encKV(enc, k, v)
		}
	}
}
