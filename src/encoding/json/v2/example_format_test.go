// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jsonv2 && goexperiment.jsonformat

package json_test

import (
	"encoding/json/jsontext"
	"encoding/json/v2"
	"fmt"
	"log"
	"math"
	"time"
)

// The "format" tag option can be used to alter the formatting of certain types.
func Example_formatFlags() {
	value := struct {
		BytesBase64     []byte         `json:",format:base64"`
		BytesHex        [8]byte        `json:",format:hex"`
		BytesArray      []byte         `json:",format:array"`
		FloatNonFinite  float64        `json:",format:nonfinite"`
		MapEmitNull     map[string]any `json:",format:emitnull"`
		SliceEmitNull   []any          `json:",format:emitnull"`
		TimeDateOnly    time.Time      `json:",format:'2006-01-02'"`
		TimeUnixSec     time.Time      `json:",format:unix"`
		DurationSecs    time.Duration  `json:",format:sec"`
		DurationNanos   time.Duration  `json:",format:nano"`
		DurationISO8601 time.Duration  `json:",format:iso8601"`
	}{
		BytesBase64:     []byte{0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef},
		BytesHex:        [8]byte{0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef},
		BytesArray:      []byte{0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef},
		FloatNonFinite:  math.NaN(),
		MapEmitNull:     nil,
		SliceEmitNull:   nil,
		TimeDateOnly:    time.Date(2000, 1, 1, 0, 0, 0, 0, time.UTC),
		TimeUnixSec:     time.Date(2000, 1, 1, 0, 0, 0, 0, time.UTC),
		DurationSecs:    12*time.Hour + 34*time.Minute + 56*time.Second + 7*time.Millisecond + 8*time.Microsecond + 9*time.Nanosecond,
		DurationNanos:   12*time.Hour + 34*time.Minute + 56*time.Second + 7*time.Millisecond + 8*time.Microsecond + 9*time.Nanosecond,
		DurationISO8601: 12*time.Hour + 34*time.Minute + 56*time.Second + 7*time.Millisecond + 8*time.Microsecond + 9*time.Nanosecond,
	}

	b, err := json.Marshal(&value)
	if err != nil {
		log.Fatal(err)
	}
	(*jsontext.Value)(&b).Indent() // indent for readability
	fmt.Println(string(b))

	// Output:
	// {
	// 	"BytesBase64": "ASNFZ4mrze8=",
	// 	"BytesHex": "0123456789abcdef",
	// 	"BytesArray": [
	// 		1,
	// 		35,
	// 		69,
	// 		103,
	// 		137,
	// 		171,
	// 		205,
	// 		239
	// 	],
	// 	"FloatNonFinite": "NaN",
	// 	"MapEmitNull": null,
	// 	"SliceEmitNull": null,
	//	"TimeDateOnly": "2000-01-01",
	//	"TimeUnixSec": 946684800,
	//	"DurationSecs": 45296.007008009,
	//	"DurationNanos": 45296007008009,
	//	"DurationISO8601": "PT12H34M56.007008009S"
	// }
}
