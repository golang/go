// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http3

import (
	"golang.org/x/net/internal/httpcommon"
)

type qpackEncoder struct {
	// The encoder has no state for now,
	// but that'll change once we add dynamic table support.
	//
	// TODO: dynamic table support.
}

func (qe *qpackEncoder) init() {
	staticTableOnce.Do(initStaticTableMaps)
}

// encode encodes a list of headers into a QPACK encoded field section.
//
// The headers func must produce the same headers on repeated calls,
// although the order may vary.
func (qe *qpackEncoder) encode(headers func(func(itype indexType, name, value string))) []byte {
	// Encoded Field Section prefix.
	//
	// We don't yet use the dynamic table, so both values here are zero.
	var b []byte
	b = appendPrefixedInt(b, 0, 8, 0) // Required Insert Count
	b = appendPrefixedInt(b, 0, 7, 0) // Delta Base

	headers(func(itype indexType, name, value string) {
		// Technically, it is the responsibility of the protocol using HTTP/3
		// to ensure that all field names are already in lowercase. However,
		// this QPACK implementation is solely used by and live in the http3
		// package. So, we might as well do the lowercasing here to make sure
		// we do not miss any callsites or need to create yet another struct
		// wrapping the qpackEncoder.
		name, ascii := httpcommon.LowerHeader(name)
		// Skip writing invalid headers. Per RFC 9114 section 4.2: "Field
		// names are strings containing a subset of ASCII characters."
		if !ascii {
			return
		}
		if itype == mayIndex {
			if i, ok := staticTableByNameValue[tableEntry{name, value}]; ok {
				b = appendIndexedFieldLine(b, staticTable, i)
				return
			}
		}
		if i, ok := staticTableByName[name]; ok {
			b = appendLiteralFieldLineWithNameReference(b, staticTable, itype, i, value)
		} else {
			b = appendLiteralFieldLineWithLiteralName(b, itype, name, value)
		}
	})

	return b
}
