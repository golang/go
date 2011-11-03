// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http

import (
	"bytes"
	"io"
	"io/ioutil"
)

// One of the copies, say from b to r2, could be avoided by using a more
// elaborate trick where the other copy is made during Request/Response.Write.
// This would complicate things too much, given that these functions are for
// debugging only.
func drainBody(b io.ReadCloser) (r1, r2 io.ReadCloser, err error) {
	var buf bytes.Buffer
	if _, err = buf.ReadFrom(b); err != nil {
		return nil, nil, err
	}
	if err = b.Close(); err != nil {
		return nil, nil, err
	}
	return ioutil.NopCloser(&buf), ioutil.NopCloser(bytes.NewBuffer(buf.Bytes())), nil
}

// DumpRequest returns the wire representation of req,
// optionally including the request body, for debugging.
// DumpRequest is semantically a no-op, but in order to
// dump the body, it reads the body data into memory and
// changes req.Body to refer to the in-memory copy.
// The documentation for Request.Write details which fields
// of req are used.
func DumpRequest(req *Request, body bool) (dump []byte, err error) {
	var b bytes.Buffer
	save := req.Body
	if !body || req.Body == nil {
		req.Body = nil
	} else {
		save, req.Body, err = drainBody(req.Body)
		if err != nil {
			return
		}
	}
	err = req.dumpWrite(&b)
	req.Body = save
	if err != nil {
		return
	}
	dump = b.Bytes()
	return
}

// DumpResponse is like DumpRequest but dumps a response.
func DumpResponse(resp *Response, body bool) (dump []byte, err error) {
	var b bytes.Buffer
	save := resp.Body
	savecl := resp.ContentLength
	if !body || resp.Body == nil {
		resp.Body = nil
		resp.ContentLength = 0
	} else {
		save, resp.Body, err = drainBody(resp.Body)
		if err != nil {
			return
		}
	}
	err = resp.Write(&b)
	resp.Body = save
	resp.ContentLength = savecl
	if err != nil {
		return
	}
	dump = b.Bytes()
	return
}
