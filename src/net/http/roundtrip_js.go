// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build js && wasm

package http

import (
	"errors"
	"fmt"
	"io"
	"net/http/internal/ascii"
	"strconv"
	"strings"
	"syscall/js"
)

var uint8Array = js.Global().Get("Uint8Array")

// jsFetchMode is a Request.Header map key that, if present,
// signals that the map entry is actually an option to the Fetch API mode setting.
// Valid values are: "cors", "no-cors", "same-origin", "navigate"
// The default is "same-origin".
//
// Reference: https://developer.mozilla.org/en-US/docs/Web/API/WindowOrWorkerGlobalScope/fetch#Parameters
const jsFetchMode = "js.fetch:mode"

// jsFetchCreds is a Request.Header map key that, if present,
// signals that the map entry is actually an option to the Fetch API credentials setting.
// Valid values are: "omit", "same-origin", "include"
// The default is "same-origin".
//
// Reference: https://developer.mozilla.org/en-US/docs/Web/API/WindowOrWorkerGlobalScope/fetch#Parameters
const jsFetchCreds = "js.fetch:credentials"

// jsFetchRedirect is a Request.Header map key that, if present,
// signals that the map entry is actually an option to the Fetch API redirect setting.
// Valid values are: "follow", "error", "manual"
// The default is "follow".
//
// Reference: https://developer.mozilla.org/en-US/docs/Web/API/WindowOrWorkerGlobalScope/fetch#Parameters
const jsFetchRedirect = "js.fetch:redirect"

// jsFetchMissing will be true if the Fetch API is not present in
// the browser globals.
var jsFetchMissing = js.Global().Get("fetch").IsUndefined()

// jsFetchDisabled controls whether the use of Fetch API is disabled.
// It's set to true when we detect we're running in Node.js, so that
// RoundTrip ends up talking over the same fake network the HTTP servers
// currently use in various tests and examples. See go.dev/issue/57613.
//
// TODO(go.dev/issue/60810): See if it's viable to test the Fetch API
// code path.
var jsFetchDisabled = js.Global().Get("process").Type() == js.TypeObject &&
	strings.HasPrefix(js.Global().Get("process").Get("argv0").String(), "node")

// RoundTrip implements the [RoundTripper] interface using the WHATWG Fetch API.
func (t *Transport) RoundTrip(req *Request) (*Response, error) {
	// The Transport has a documented contract that states that if the DialContext or
	// DialTLSContext functions are set, they will be used to set up the connections.
	// If they aren't set then the documented contract is to use Dial or DialTLS, even
	// though they are deprecated. Therefore, if any of these are set, we should obey
	// the contract and dial using the regular round-trip instead. Otherwise, we'll try
	// to fall back on the Fetch API, unless it's not available.
	if t.Dial != nil || t.DialContext != nil || t.DialTLS != nil || t.DialTLSContext != nil || jsFetchMissing || jsFetchDisabled {
		return t.roundTrip(req)
	}

	ac := js.Global().Get("AbortController")
	if !ac.IsUndefined() {
		// Some browsers that support WASM don't necessarily support
		// the AbortController. See
		// https://developer.mozilla.org/en-US/docs/Web/API/AbortController#Browser_compatibility.
		ac = ac.New()
	}

	opt := js.Global().Get("Object").New()
	// See https://developer.mozilla.org/en-US/docs/Web/API/WindowOrWorkerGlobalScope/fetch
	// for options available.
	opt.Set("method", req.Method)
	opt.Set("credentials", "same-origin")
	if h := req.Header.Get(jsFetchCreds); h != "" {
		opt.Set("credentials", h)
		req.Header.Del(jsFetchCreds)
	}
	if h := req.Header.Get(jsFetchMode); h != "" {
		opt.Set("mode", h)
		req.Header.Del(jsFetchMode)
	}
	if h := req.Header.Get(jsFetchRedirect); h != "" {
		opt.Set("redirect", h)
		req.Header.Del(jsFetchRedirect)
	}
	if !ac.IsUndefined() {
		opt.Set("signal", ac.Get("signal"))
	}
	headers := js.Global().Get("Headers").New()
	for key, values := range req.Header {
		for _, value := range values {
			headers.Call("append", key, value)
		}
	}
	opt.Set("headers", headers)

	if req.Body != nil {
		// TODO(johanbrandhorst): Stream request body when possible.
		// See https://bugs.chromium.org/p/chromium/issues/detail?id=688906 for Blink issue.
		// See https://bugzilla.mozilla.org/show_bug.cgi?id=1387483 for Firefox issue.
		// See https://github.com/web-platform-tests/wpt/issues/7693 for WHATWG tests issue.
		// See https://developer.mozilla.org/en-US/docs/Web/API/Streams_API for more details on the Streams API
		// and browser support.
		// NOTE(haruyama480): Ensure HTTP/1 fallback exists.
		// See https://go.dev/issue/61889 for discussion.
		body, err := io.ReadAll(req.Body)
		if err != nil {
			req.Body.Close() // RoundTrip must always close the body, including on errors.
			return nil, err
		}
		req.Body.Close()
		if len(body) != 0 {
			buf := uint8Array.New(len(body))
			js.CopyBytesToJS(buf, body)
			opt.Set("body", buf)
		}
	}

	fetchPromise := js.Global().Call("fetch", req.URL.String(), opt)
	var (
		respCh           = make(chan *Response, 1)
		errCh            = make(chan error, 1)
		success, failure js.Func
	)
	success = js.FuncOf(func(this js.Value, args []js.Value) any {
		success.Release()
		failure.Release()

		result := args[0]
		header := Header{}
		// https://developer.mozilla.org/en-US/docs/Web/API/Headers/entries
		headersIt := result.Get("headers").Call("entries")
		for {
			n := headersIt.Call("next")
			if n.Get("done").Bool() {
				break
			}
			pair := n.Get("value")
			key, value := pair.Index(0).String(), pair.Index(1).String()
			ck := CanonicalHeaderKey(key)
			header[ck] = append(header[ck], value)
		}

		contentLength := int64(0)
		clHeader := header.Get("Content-Length")
		switch {
		case clHeader != "":
			cl, err := strconv.ParseInt(clHeader, 10, 64)
			if err != nil {
				errCh <- fmt.Errorf("net/http: ill-formed Content-Length header: %v", err)
				return nil
			}
			if cl < 0 {
				// Content-Length values less than 0 are invalid.
				// See: https://datatracker.ietf.org/doc/html/rfc2616/#section-14.13
				errCh <- fmt.Errorf("net/http: invalid Content-Length header: %q", clHeader)
				return nil
			}
			contentLength = cl
		default:
			// If the response length is not declared, set it to -1.
			contentLength = -1
		}

		b := result.Get("body")
		var body io.ReadCloser
		// The body is undefined when the browser does not support streaming response bodies (Firefox),
		// and null in certain error cases, i.e. when the request is blocked because of CORS settings.
		if !b.IsUndefined() && !b.IsNull() {
			body = &streamReader{stream: b.Call("getReader")}
		} else {
			// Fall back to using ArrayBuffer
			// https://developer.mozilla.org/en-US/docs/Web/API/Body/arrayBuffer
			body = &arrayReader{arrayPromise: result.Call("arrayBuffer")}
		}

		code := result.Get("status").Int()

		uncompressed := false
		if ascii.EqualFold(header.Get("Content-Encoding"), "gzip") {
			// The fetch api will decode the gzip, but Content-Encoding not be deleted.
			header.Del("Content-Encoding")
			header.Del("Content-Length")
			contentLength = -1
			uncompressed = true
		}

		respCh <- &Response{
			Status:        fmt.Sprintf("%d %s", code, StatusText(code)),
			StatusCode:    code,
			Header:        header,
			ContentLength: contentLength,
			Uncompressed:  uncompressed,
			Body:          body,
			Request:       req,
		}

		return nil
	})
	failure = js.FuncOf(func(this js.Value, args []js.Value) any {
		success.Release()
		failure.Release()

		err := args[0]
		// The error is a JS Error type
		// https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Error
		// We can use the toString() method to get a string representation of the error.
		errMsg := err.Call("toString").String()
		// Errors can optionally contain a cause.
		if cause := err.Get("cause"); !cause.IsUndefined() {
			// The exact type of the cause is not defined,
			// but if it's another error, we can call toString() on it too.
			if !cause.Get("toString").IsUndefined() {
				errMsg += ": " + cause.Call("toString").String()
			} else if cause.Type() == js.TypeString {
				errMsg += ": " + cause.String()
			}
		}
		errCh <- fmt.Errorf("net/http: fetch() failed: %s", errMsg)
		return nil
	})

	fetchPromise.Call("then", success, failure)
	select {
	case <-req.Context().Done():
		if !ac.IsUndefined() {
			// Abort the Fetch request.
			ac.Call("abort")
		}
		return nil, req.Context().Err()
	case resp := <-respCh:
		return resp, nil
	case err := <-errCh:
		return nil, err
	}
}

var errClosed = errors.New("net/http: reader is closed")

// streamReader implements an io.ReadCloser wrapper for ReadableStream.
// See https://fetch.spec.whatwg.org/#readablestream for more information.
type streamReader struct {
	pending []byte
	stream  js.Value
	err     error // sticky read error
}

func (r *streamReader) Read(p []byte) (n int, err error) {
	if r.err != nil {
		return 0, r.err
	}
	if len(r.pending) == 0 {
		var (
			bCh   = make(chan []byte, 1)
			errCh = make(chan error, 1)
		)
		success := js.FuncOf(func(this js.Value, args []js.Value) any {
			result := args[0]
			if result.Get("done").Bool() {
				errCh <- io.EOF
				return nil
			}
			value := make([]byte, result.Get("value").Get("byteLength").Int())
			js.CopyBytesToGo(value, result.Get("value"))
			bCh <- value
			return nil
		})
		defer success.Release()
		failure := js.FuncOf(func(this js.Value, args []js.Value) any {
			// Assumes it's a TypeError. See
			// https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/TypeError
			// for more information on this type. See
			// https://streams.spec.whatwg.org/#byob-reader-read for the spec on
			// the read method.
			errCh <- errors.New(args[0].Get("message").String())
			return nil
		})
		defer failure.Release()
		r.stream.Call("read").Call("then", success, failure)
		select {
		case b := <-bCh:
			r.pending = b
		case err := <-errCh:
			r.err = err
			return 0, err
		}
	}
	n = copy(p, r.pending)
	r.pending = r.pending[n:]
	return n, nil
}

func (r *streamReader) Close() error {
	// This ignores any error returned from cancel method. So far, I did not encounter any concrete
	// situation where reporting the error is meaningful. Most users ignore error from resp.Body.Close().
	// If there's a need to report error here, it can be implemented and tested when that need comes up.
	r.stream.Call("cancel")
	if r.err == nil {
		r.err = errClosed
	}
	return nil
}

// arrayReader implements an io.ReadCloser wrapper for ArrayBuffer.
// https://developer.mozilla.org/en-US/docs/Web/API/Body/arrayBuffer.
type arrayReader struct {
	arrayPromise js.Value
	pending      []byte
	read         bool
	err          error // sticky read error
}

func (r *arrayReader) Read(p []byte) (n int, err error) {
	if r.err != nil {
		return 0, r.err
	}
	if !r.read {
		r.read = true
		var (
			bCh   = make(chan []byte, 1)
			errCh = make(chan error, 1)
		)
		success := js.FuncOf(func(this js.Value, args []js.Value) any {
			// Wrap the input ArrayBuffer with a Uint8Array
			uint8arrayWrapper := uint8Array.New(args[0])
			value := make([]byte, uint8arrayWrapper.Get("byteLength").Int())
			js.CopyBytesToGo(value, uint8arrayWrapper)
			bCh <- value
			return nil
		})
		defer success.Release()
		failure := js.FuncOf(func(this js.Value, args []js.Value) any {
			// Assumes it's a TypeError. See
			// https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/TypeError
			// for more information on this type.
			// See https://fetch.spec.whatwg.org/#concept-body-consume-body for reasons this might error.
			errCh <- errors.New(args[0].Get("message").String())
			return nil
		})
		defer failure.Release()
		r.arrayPromise.Call("then", success, failure)
		select {
		case b := <-bCh:
			r.pending = b
		case err := <-errCh:
			return 0, err
		}
	}
	if len(r.pending) == 0 {
		return 0, io.EOF
	}
	n = copy(p, r.pending)
	r.pending = r.pending[n:]
	return n, nil
}

func (r *arrayReader) Close() error {
	if r.err == nil {
		r.err = errClosed
	}
	return nil
}
