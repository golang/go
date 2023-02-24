// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This benchmark tests JSON encoding and decoding performance.

package go1

import (
	"bytes"
	"compress/bzip2"
	"encoding/base64"
	"encoding/json"
	"io"
	"testing"
)

func makeJsonBytes() []byte {
	var r io.Reader
	r = bytes.NewReader(bytes.Replace(jsonbz2_base64, []byte{'\n'}, nil, -1))
	r = base64.NewDecoder(base64.StdEncoding, r)
	r = bzip2.NewReader(r)
	b, err := io.ReadAll(r)
	if err != nil {
		panic(err)
	}
	return b
}

func makeJsonData(jsonbytes []byte) *JSONResponse {
	var v JSONResponse
	if err := json.Unmarshal(jsonbytes, &v); err != nil {
		panic(err)
	}
	return &v
}

type JSONResponse struct {
	Tree     *JSONNode `json:"tree"`
	Username string    `json:"username"`
}

type JSONNode struct {
	Name     string      `json:"name"`
	Kids     []*JSONNode `json:"kids"`
	CLWeight float64     `json:"cl_weight"`
	Touches  int         `json:"touches"`
	MinT     int64       `json:"min_t"`
	MaxT     int64       `json:"max_t"`
	MeanT    int64       `json:"mean_t"`
}

func jsondec(bytes []byte) {
	var r JSONResponse
	if err := json.Unmarshal(bytes, &r); err != nil {
		panic(err)
	}
	_ = r
}

func jsonenc(data *JSONResponse) {
	buf, err := json.Marshal(data)
	if err != nil {
		panic(err)
	}
	_ = buf
}

func BenchmarkJSONEncode(b *testing.B) {
	jsonbytes := makeJsonBytes()
	jsondata := makeJsonData(jsonbytes)
	b.ResetTimer()
	b.SetBytes(int64(len(jsonbytes)))
	for i := 0; i < b.N; i++ {
		jsonenc(jsondata)
	}
}

func BenchmarkJSONDecode(b *testing.B) {
	jsonbytes := makeJsonBytes()
	b.ResetTimer()
	b.SetBytes(int64(len(jsonbytes)))
	for i := 0; i < b.N; i++ {
		jsondec(jsonbytes)
	}
}
