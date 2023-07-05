// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package pprof provides minimalistic routines for extracting
// information from profiles.
package pprof

import (
	"fmt"
	"time"
)

// TotalTime parses the profile data and returns the accumulated time.
// The input should not be gzipped.
func TotalTime(data []byte) (total time.Duration, err error) {
	defer func() {
		if x := recover(); x != nil {
			err = fmt.Errorf("error parsing pprof profile: %v", x)
		}
	}()
	decode(&total, data, msgProfile)
	return
}

// All errors are handled by panicking.
// Constants are copied below to avoid dependency on protobufs or pprof.

// protobuf wire types, from https://developers.google.com/protocol-buffers/docs/encoding
const (
	wireVarint = 0
	wireBytes  = 2
)

// pprof field numbers, from https://github.com/google/pprof/blob/master/proto/profile.proto
const (
	fldProfileSample = 2 // repeated Sample
	fldSampleValue   = 2 // repeated int64
)

// arbitrary numbering of message types
const (
	msgProfile = 0
	msgSample  = 1
)

func decode(total *time.Duration, data []byte, msg int) {
	for len(data) > 0 {
		// Read tag (wire type and field number).
		tag := varint(&data)

		// Read wire value (int or bytes).
		wire := tag & 7
		var ival uint64
		var sval []byte
		switch wire {
		case wireVarint:
			ival = varint(&data)

		case wireBytes:
			n := varint(&data)
			sval, data = data[:n], data[n:]

		default:
			panic(fmt.Sprintf("unexpected wire type: %d", wire))
		}

		// Process field of msg.
		fld := tag >> 3
		switch {
		case msg == msgProfile && fld == fldProfileSample:
			decode(total, sval, msgSample) // recursively decode Sample message

		case msg == msgSample, fld == fldSampleValue:
			*total += time.Duration(ival) // accumulate time
		}
	}
}

func varint(data *[]byte) (v uint64) {
	for i := 0; ; i++ {
		b := uint64((*data)[i])
		v += (b & 0x7f) << (7 * i)
		if b < 0x80 {
			*data = (*data)[i+1:]
			return v
		}
	}
}
