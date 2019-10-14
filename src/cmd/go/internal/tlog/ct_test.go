// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tlog

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/url"
	"os"
	"testing"
)

func TestCertificateTransparency(t *testing.T) {
	// Test that we can verify actual Certificate Transparency proofs.
	// (The other tests check that we can verify our own proofs;
	// this is a test that the two are compatible.)

	if testing.Short() {
		t.Skip("skipping in -short mode")
	}

	var root ctTree
	httpGET(t, "http://ct.googleapis.com/logs/argon2020/ct/v1/get-sth", &root)

	var leaf ctEntries
	httpGET(t, "http://ct.googleapis.com/logs/argon2020/ct/v1/get-entries?start=10000&end=10000", &leaf)
	hash := RecordHash(leaf.Entries[0].Data)

	var rp ctRecordProof
	httpGET(t, "http://ct.googleapis.com/logs/argon2020/ct/v1/get-proof-by-hash?tree_size="+fmt.Sprint(root.Size)+"&hash="+url.QueryEscape(hash.String()), &rp)

	err := CheckRecord(rp.Proof, root.Size, root.Hash, 10000, hash)
	if err != nil {
		t.Fatal(err)
	}

	var tp ctTreeProof
	httpGET(t, "http://ct.googleapis.com/logs/argon2020/ct/v1/get-sth-consistency?first=3654490&second="+fmt.Sprint(root.Size), &tp)

	oh, _ := ParseHash("AuIZ5V6sDUj1vn3Y1K85oOaQ7y+FJJKtyRTl1edIKBQ=")
	err = CheckTree(tp.Proof, root.Size, root.Hash, 3654490, oh)
	if err != nil {
		t.Fatal(err)
	}
}

type ctTree struct {
	Size int64 `json:"tree_size"`
	Hash Hash  `json:"sha256_root_hash"`
}

type ctEntries struct {
	Entries []*ctEntry
}

type ctEntry struct {
	Data []byte `json:"leaf_input"`
}

type ctRecordProof struct {
	Index int64       `json:"leaf_index"`
	Proof RecordProof `json:"audit_path"`
}

type ctTreeProof struct {
	Proof TreeProof `json:"consistency"`
}

func httpGET(t *testing.T, url string, targ interface{}) {
	if testing.Verbose() {
		println()
		println(url)
	}
	resp, err := http.Get(url)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	data, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	if testing.Verbose() {
		os.Stdout.Write(data)
	}
	err = json.Unmarshal(data, targ)
	if err != nil {
		println(url)
		os.Stdout.Write(data)
		t.Fatal(err)
	}
}
