// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This benchmark tests gob encoding and decoding performance.

package go1

import (
	"bytes"
	"encoding/gob"
	"encoding/json"
	"io/ioutil"
	"log"
	"reflect"
	"testing"
)

var (
	gobbytes []byte
	gobdata  *JSONResponse
)

func gobinit() {
	// gobinit is called after json's init,
	// because it uses jsondata.
	gobdata = gobResponse(&jsondata)

	var buf bytes.Buffer
	if err := gob.NewEncoder(&buf).Encode(gobdata); err != nil {
		panic(err)
	}
	gobbytes = buf.Bytes()

	var r JSONResponse
	if err := gob.NewDecoder(bytes.NewBuffer(gobbytes)).Decode(&r); err != nil {
		panic(err)
	}
	if !reflect.DeepEqual(gobdata, &r) {
		log.Printf("%v\n%v", jsondata, r)
		b, _ := json.Marshal(&jsondata)
		br, _ := json.Marshal(&r)
		log.Printf("%s\n%s\n", b, br)
		panic("gob: encode+decode lost data")
	}
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

func gobdec() {
	if gobbytes == nil {
		panic("gobdata not initialized")
	}
	var r JSONResponse
	if err := gob.NewDecoder(bytes.NewBuffer(gobbytes)).Decode(&r); err != nil {
		panic(err)
	}
	_ = r
}

func gobenc() {
	if err := gob.NewEncoder(ioutil.Discard).Encode(&gobdata); err != nil {
		panic(err)
	}
}

func BenchmarkGobDecode(b *testing.B) {
	b.SetBytes(int64(len(gobbytes)))
	for i := 0; i < b.N; i++ {
		gobdec()
	}
}

func BenchmarkGobEncode(b *testing.B) {
	b.SetBytes(int64(len(gobbytes)))
	for i := 0; i < b.N; i++ {
		gobenc()
	}
}
