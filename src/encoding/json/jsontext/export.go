// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jsonv2

package jsontext

import (
	"io"

	"encoding/json/internal"
)

// TODO(https://go.dev/issue/73435): Remove the Internal symbol.
//
// The Go language lacks a 3rd category of visibility where
// certain symbols can only be referenced from within the same module.
// One solution to this is to put the entirety of a package with
// both public and module-only symbols exposed as an internal package.
// A separate, public package can re-export all of the public symbols
// via type aliases and thus ensuring module-only symbols cannot be
// referenced by the end-user. While this works, it unfortunately
// leads to a poor user experience since the Go pkgsite is unable to
// forward the documentation for symbols like methods and fields.
// We need to improve the pkgsite experience before we can delete
// the Internal symbol.

// Internal is for internal use only.
// This is exempt from the Go compatibility agreement.
var Internal exporter

type exporter struct{}

// Export exposes internal functionality from "jsontext" to "json".
// This cannot be dynamically called by other packages since
// they cannot obtain a reference to the internal.AllowInternalUse value.
func (exporter) Export(p *internal.NotForPublicUse) export {
	if p != &internal.AllowInternalUse {
		panic("unauthorized call to Export")
	}
	return export{}
}

// The export type exposes functionality to packages with visibility to
// the internal.AllowInternalUse variable. The "json" package uses this
// to modify low-level state in the Encoder and Decoder types.
// It mutates the state directly instead of calling ReadToken or WriteToken
// since this is more performant. The public APIs need to track state to ensure
// that users are constructing a valid JSON value, but the "json" implementation
// guarantees that it emits valid JSON by the structure of the code itself.
type export struct{}

// Encoder returns a pointer to the underlying encoderState.
func (export) Encoder(e *Encoder) *encoderState { return &e.s }

// Decoder returns a pointer to the underlying decoderState.
func (export) Decoder(d *Decoder) *decoderState { return &d.s }

func (export) GetBufferedEncoder(o ...Options) *Encoder {
	return getBufferedEncoder(o...)
}
func (export) PutBufferedEncoder(e *Encoder) {
	putBufferedEncoder(e)
}

func (export) GetStreamingEncoder(w io.Writer, o ...Options) *Encoder {
	return getStreamingEncoder(w, o...)
}
func (export) PutStreamingEncoder(e *Encoder) {
	putStreamingEncoder(e)
}

func (export) GetBufferedDecoder(b []byte, o ...Options) *Decoder {
	return getBufferedDecoder(b, o...)
}
func (export) PutBufferedDecoder(d *Decoder) {
	putBufferedDecoder(d)
}

func (export) GetStreamingDecoder(r io.Reader, o ...Options) *Decoder {
	return getStreamingDecoder(r, o...)
}
func (export) PutStreamingDecoder(d *Decoder) {
	putStreamingDecoder(d)
}

func (export) IsIOError(err error) bool {
	_, ok := err.(*ioError)
	return ok
}
