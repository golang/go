// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This benchmark tests gob encoding and decoding performance.

package go1

import (
	"bytes"
	"encoding/gob"
	"encoding/json"
	"io"
	"log"
	"reflect"
	"testing"
)

func makeGob(jsondata *JSONResponse) (data *JSONResponse, b []byte) {
	data = gobResponse(jsondata)

	var buf bytes.Buffer
	if err := gob.NewEncoder(&buf).Encode(data); err != nil {
		panic(err)
	}
	b = buf.Bytes()

	var r JSONResponse
	if err := gob.NewDecoder(bytes.NewBuffer(b)).Decode(&r); err != nil {
		panic(err)
	}
	if !reflect.DeepEqual(data, &r) {
		log.Printf("%v\n%v", jsondata, r)
		b, _ := json.Marshal(&jsondata)
		br, _ := json.Marshal(&r)
		log.Printf("%s\n%s\n", b, br)
		panic("gob: encode+decode lost data")
	}

	return
}

// gob turns [] into null, so make a copy of the data structure like that
func gobResponse(r *JSONResponse) *JSONResponse {
	return &JSONResponse{gobNode(r.Tree), r.Username}
}

func gobNode(n *JSONNode) *JSONNode {
	n1 := new(JSONNode)
	*n1 = *n
	if len(n1.Kids) == 0 {
		n1.Kids = nil
	} else {
		for i, k := range n1.Kids {
			n1.Kids[i] = gobNode(k)
		}
	}
	return n1
}

func gobdec(b []byte) {
	var r JSONResponse
	if err := gob.NewDecoder(bytes.NewBuffer(b)).Decode(&r); err != nil {
		panic(err)
	}
	_ = r
}

func gobenc(data *JSONResponse) {
	if err := gob.NewEncoder(io.Discard).Encode(data); err != nil {
		panic(err)
	}
}

func BenchmarkGobDecode(b *testing.B) {
	jsonbytes := makeJsonBytes()
	jsondata := makeJsonData(jsonbytes)
	_, bytes := makeGob(jsondata)
	b.ResetTimer()
	b.SetBytes(int64(len(bytes)))
	for i := 0; i < b.N; i++ {
		gobdec(bytes)
	}
}

func BenchmarkGobEncode(b *testing.B) {
	jsonbytes := makeJsonBytes()
	jsondata := makeJsonData(jsonbytes)
	data, bytes := makeGob(jsondata)
	b.ResetTimer()
	b.SetBytes(int64(len(bytes)))
	for i := 0; i < b.N; i++ {
		gobenc(data)
	}
}
