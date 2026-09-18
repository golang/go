// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http3

import (
	"errors"
	"math/bits"
)

type qpackDecoder struct {
	// The decoder has no state for now,
	// but that'll change once we add dynamic table support.
	//
	// TODO: dynamic table support.
}

func (qd *qpackDecoder) decode(st *stream, f func(itype indexType, name, value string) error) error {
	// Encoded Field Section prefix.

	// We set SETTINGS_QPACK_MAX_TABLE_CAPACITY to 0,
	// so the Required Insert Count must be 0.
	_, requiredInsertCount, err := st.readPrefixedInt(8)
	if err != nil {
		return err
	}
	if requiredInsertCount != 0 {
		return errQPACKDecompressionFailed
	}

	// Delta Base. We don't use the dynamic table yet, so this may be ignored.
	_, _, err = st.readPrefixedInt(7)
	if err != nil {
		return err
	}

	sawNonPseudo := false
	for st.lim > 0 {
		firstByte, err := st.ReadByte()
		if err != nil {
			return err
		}
		var name, value string
		var itype indexType
		switch bits.LeadingZeros8(firstByte) {
		case 0:
			// Indexed Field Line
			itype, name, value, err = st.decodeIndexedFieldLine(firstByte)
		case 1:
			// Literal Field Line With Name Reference
			itype, name, value, err = st.decodeLiteralFieldLineWithNameReference(firstByte)
		case 2:
			// Literal Field Line with Literal Name
			itype, name, value, err = st.decodeLiteralFieldLineWithLiteralName(firstByte)
		case 3:
			// Indexed Field Line With Post-Base Index
			err = errors.New("dynamic table is not supported yet")
		case 4:
			// Indexed Field Line With Post-Base Name Reference
			err = errors.New("dynamic table is not supported yet")
		}
		if err != nil {
			return err
		}
		if len(name) == 0 {
			return errH3MessageError
		}
		if name[0] == ':' {
			if sawNonPseudo {
				return errH3MessageError
			}
		} else {
			sawNonPseudo = true
		}
		if err := f(itype, name, value); err != nil {
			return err
		}
	}
	return nil
}
