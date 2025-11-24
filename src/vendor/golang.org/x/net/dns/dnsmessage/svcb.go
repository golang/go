// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package dnsmessage

import (
	"slices"
)

// An SVCBResource is an SVCB Resource record.
type SVCBResource struct {
	Priority uint16
	Target   Name
	Params   []SVCParam // Must be in strict increasing order by Key.
}

func (r *SVCBResource) realType() Type {
	return TypeSVCB
}

// GoString implements fmt.GoStringer.GoString.
func (r *SVCBResource) GoString() string {
	b := []byte("dnsmessage.SVCBResource{" +
		"Priority: " + printUint16(r.Priority) + ", " +
		"Target: " + r.Target.GoString() + ", " +
		"Params: []dnsmessage.SVCParam{")
	if len(r.Params) > 0 {
		b = append(b, r.Params[0].GoString()...)
		for _, p := range r.Params[1:] {
			b = append(b, ", "+p.GoString()...)
		}
	}
	b = append(b, "}}"...)
	return string(b)
}

// An HTTPSResource is an HTTPS Resource record.
// It has the same format as the SVCB record.
type HTTPSResource struct {
	// Alias for SVCB resource record.
	SVCBResource
}

func (r *HTTPSResource) realType() Type {
	return TypeHTTPS
}

// GoString implements fmt.GoStringer.GoString.
func (r *HTTPSResource) GoString() string {
	return "dnsmessage.HTTPSResource{SVCBResource: " + r.SVCBResource.GoString() + "}"
}

// GetParam returns a parameter value by key.
func (r *SVCBResource) GetParam(key SVCParamKey) (value []byte, ok bool) {
	for i := range r.Params {
		if r.Params[i].Key == key {
			return r.Params[i].Value, true
		}
		if r.Params[i].Key > key {
			break
		}
	}
	return nil, false
}

// SetParam sets a parameter value by key.
// The Params list is kept sorted by key.
func (r *SVCBResource) SetParam(key SVCParamKey, value []byte) {
	i := 0
	for i < len(r.Params) {
		if r.Params[i].Key >= key {
			break
		}
		i++
	}

	if i < len(r.Params) && r.Params[i].Key == key {
		r.Params[i].Value = value
		return
	}

	r.Params = slices.Insert(r.Params, i, SVCParam{Key: key, Value: value})
}

// DeleteParam deletes a parameter by key.
// It returns true if the parameter was present.
func (r *SVCBResource) DeleteParam(key SVCParamKey) bool {
	for i := range r.Params {
		if r.Params[i].Key == key {
			r.Params = slices.Delete(r.Params, i, i+1)
			return true
		}
		if r.Params[i].Key > key {
			break
		}
	}
	return false
}

// A SVCParam is a service parameter.
type SVCParam struct {
	Key   SVCParamKey
	Value []byte
}

// GoString implements fmt.GoStringer.GoString.
func (p SVCParam) GoString() string {
	return "dnsmessage.SVCParam{" +
		"Key: " + p.Key.GoString() + ", " +
		"Value: []byte{" + printByteSlice(p.Value) + "}}"
}

// A SVCParamKey is a key for a service parameter.
type SVCParamKey uint16

// Values defined at https://www.iana.org/assignments/dns-svcb/dns-svcb.xhtml#dns-svcparamkeys.
const (
	SVCParamMandatory          SVCParamKey = 0
	SVCParamALPN               SVCParamKey = 1
	SVCParamNoDefaultALPN      SVCParamKey = 2
	SVCParamPort               SVCParamKey = 3
	SVCParamIPv4Hint           SVCParamKey = 4
	SVCParamECH                SVCParamKey = 5
	SVCParamIPv6Hint           SVCParamKey = 6
	SVCParamDOHPath            SVCParamKey = 7
	SVCParamOHTTP              SVCParamKey = 8
	SVCParamTLSSupportedGroups SVCParamKey = 9
)

var svcParamKeyNames = map[SVCParamKey]string{
	SVCParamMandatory:          "Mandatory",
	SVCParamALPN:               "ALPN",
	SVCParamNoDefaultALPN:      "NoDefaultALPN",
	SVCParamPort:               "Port",
	SVCParamIPv4Hint:           "IPv4Hint",
	SVCParamECH:                "ECH",
	SVCParamIPv6Hint:           "IPv6Hint",
	SVCParamDOHPath:            "DOHPath",
	SVCParamOHTTP:              "OHTTP",
	SVCParamTLSSupportedGroups: "TLSSupportedGroups",
}

// String implements fmt.Stringer.String.
func (k SVCParamKey) String() string {
	if n, ok := svcParamKeyNames[k]; ok {
		return n
	}
	return printUint16(uint16(k))
}

// GoString implements fmt.GoStringer.GoString.
func (k SVCParamKey) GoString() string {
	if n, ok := svcParamKeyNames[k]; ok {
		return "dnsmessage.SVCParam" + n
	}
	return printUint16(uint16(k))
}

func (r *SVCBResource) pack(msg []byte, _ map[string]uint16, _ int) ([]byte, error) {
	oldMsg := msg
	msg = packUint16(msg, r.Priority)
	// https://datatracker.ietf.org/doc/html/rfc3597#section-4 prohibits name
	// compression for RR types that are not "well-known".
	// https://datatracker.ietf.org/doc/html/rfc9460#section-2.2 explicitly states that
	// compression of the Target is prohibited, following RFC 3597.
	msg, err := r.Target.pack(msg, nil, 0)
	if err != nil {
		return oldMsg, &nestedError{"SVCBResource.Target", err}
	}
	var previousKey SVCParamKey
	for i, param := range r.Params {
		if i > 0 && param.Key <= previousKey {
			return oldMsg, &nestedError{"SVCBResource.Params", errParamOutOfOrder}
		}
		if len(param.Value) > (1<<16)-1 {
			return oldMsg, &nestedError{"SVCBResource.Params", errTooLongSVCBValue}
		}
		msg = packUint16(msg, uint16(param.Key))
		msg = packUint16(msg, uint16(len(param.Value)))
		msg = append(msg, param.Value...)
	}
	return msg, nil
}

func unpackSVCBResource(msg []byte, off int, length uint16) (SVCBResource, error) {
	// Wire format reference: https://www.rfc-editor.org/rfc/rfc9460.html#section-2.2.
	r := SVCBResource{}
	paramsOff := off
	bodyEnd := off + int(length)

	var err error
	if r.Priority, paramsOff, err = unpackUint16(msg, paramsOff); err != nil {
		return SVCBResource{}, &nestedError{"Priority", err}
	}

	if paramsOff, err = r.Target.unpack(msg, paramsOff); err != nil {
		return SVCBResource{}, &nestedError{"Target", err}
	}

	// Two-pass parsing to avoid allocations.
	// First, count the number of params.
	n := 0
	var totalValueLen uint16
	off = paramsOff
	var previousKey uint16
	for off < bodyEnd {
		var key, len uint16
		if key, off, err = unpackUint16(msg, off); err != nil {
			return SVCBResource{}, &nestedError{"Params key", err}
		}
		if n > 0 && key <= previousKey {
			// As per https://www.rfc-editor.org/rfc/rfc9460.html#section-2.2, clients MUST
			// consider the RR malformed if the SvcParamKeys are not in strictly increasing numeric order
			return SVCBResource{}, &nestedError{"Params", errParamOutOfOrder}
		}
		if len, off, err = unpackUint16(msg, off); err != nil {
			return SVCBResource{}, &nestedError{"Params value length", err}
		}
		if off+int(len) > bodyEnd {
			return SVCBResource{}, errResourceLen
		}
		totalValueLen += len
		off += int(len)
		n++
	}
	if off != bodyEnd {
		return SVCBResource{}, errResourceLen
	}

	// Second, fill in the params.
	r.Params = make([]SVCParam, n)
	// valuesBuf is used to hold all param values to reduce allocations.
	// Each param's Value slice will point into this buffer.
	valuesBuf := make([]byte, totalValueLen)
	off = paramsOff
	for i := 0; i < n; i++ {
		p := &r.Params[i]
		var key, len uint16
		if key, off, err = unpackUint16(msg, off); err != nil {
			return SVCBResource{}, &nestedError{"param key", err}
		}
		p.Key = SVCParamKey(key)
		if len, off, err = unpackUint16(msg, off); err != nil {
			return SVCBResource{}, &nestedError{"param length", err}
		}
		if copy(valuesBuf, msg[off:off+int(len)]) != int(len) {
			return SVCBResource{}, &nestedError{"param value", errCalcLen}
		}
		p.Value = valuesBuf[:len:len]
		valuesBuf = valuesBuf[len:]
		off += int(len)
	}

	return r, nil
}

// genericSVCBResource parses a single Resource Record compatible with SVCB.
func (p *Parser) genericSVCBResource(svcbType Type) (SVCBResource, error) {
	if !p.resHeaderValid || p.resHeaderType != svcbType {
		return SVCBResource{}, ErrNotStarted
	}
	r, err := unpackSVCBResource(p.msg, p.off, p.resHeaderLength)
	if err != nil {
		return SVCBResource{}, err
	}
	p.off += int(p.resHeaderLength)
	p.resHeaderValid = false
	p.index++
	return r, nil
}

// SVCBResource parses a single SVCBResource.
//
// One of the XXXHeader methods must have been called before calling this
// method.
func (p *Parser) SVCBResource() (SVCBResource, error) {
	return p.genericSVCBResource(TypeSVCB)
}

// HTTPSResource parses a single HTTPSResource.
//
// One of the XXXHeader methods must have been called before calling this
// method.
func (p *Parser) HTTPSResource() (HTTPSResource, error) {
	svcb, err := p.genericSVCBResource(TypeHTTPS)
	if err != nil {
		return HTTPSResource{}, err
	}
	return HTTPSResource{svcb}, nil
}

// genericSVCBResource is the generic implementation for adding SVCB-like resources.
func (b *Builder) genericSVCBResource(h ResourceHeader, r SVCBResource) error {
	if err := b.checkResourceSection(); err != nil {
		return err
	}
	msg, lenOff, err := h.pack(b.msg, b.compression, b.start)
	if err != nil {
		return &nestedError{"ResourceHeader", err}
	}
	preLen := len(msg)
	if msg, err = r.pack(msg, b.compression, b.start); err != nil {
		return &nestedError{"ResourceBody", err}
	}
	if err := h.fixLen(msg, lenOff, preLen); err != nil {
		return err
	}
	if err := b.incrementSectionCount(); err != nil {
		return err
	}
	b.msg = msg
	return nil
}

// SVCBResource adds a single SVCBResource.
func (b *Builder) SVCBResource(h ResourceHeader, r SVCBResource) error {
	h.Type = r.realType()
	return b.genericSVCBResource(h, r)
}

// HTTPSResource adds a single HTTPSResource.
func (b *Builder) HTTPSResource(h ResourceHeader, r HTTPSResource) error {
	h.Type = r.realType()
	return b.genericSVCBResource(h, r.SVCBResource)
}
