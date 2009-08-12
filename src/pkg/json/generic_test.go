// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package json

import (
	"testing";
)

var jsontests = []string {
	`null`,
	`true`,
	`false`,
	`"abc"`,
	`123`,
	`0.1`,
	`1e-10`,
	`[]`,
	`[1,2,3,4]`,
	`[1,2,"abc",null,true,false]`,
	`{}`,
	`{"a":1}`,
}

func TestJson(t *testing.T) {
	for i := 0; i < len(jsontests); i++ {
		val, ok, errtok := StringToJson(jsontests[i]);
		if !ok {
			t.Errorf("StringToJson(%#q) => error near %v", jsontests[i], errtok);
			continue;
		}
		str := JsonToString(val);
		if str != jsontests[i] {
			t.Errorf("JsonToString(StringToJson(%#q)) = %#q", jsontests[i], str);
			continue;
		}
	}
}

func TestJsonMap(t *testing.T) {
	values := make(map[string]Json);
	mapstr := "{";
	for i := 0; i < len(jsontests); i++ {
		val, ok, errtok := StringToJson(jsontests[i]);
		if !ok {
			t.Errorf("StringToJson(%#q) => error near %v", jsontests[i], errtok);
		}
		if i > 0 {
			mapstr += ",";
		}
		values[jsontests[i]] = val;
		mapstr += Quote(jsontests[i]);
		mapstr += ":";
		mapstr += JsonToString(val);
	}
	mapstr += "}";

	mapv, ok, errtok := StringToJson(mapstr);
	if !ok {
		t.Fatalf("StringToJson(%#q) => error near %v", mapstr, errtok);
	}
	if mapv == nil {
		t.Fatalf("StringToJson(%#q) => nil, %v, %v", mapstr, ok, errtok);
	}
	if cnt := mapv.Len(); cnt != len(jsontests) {
		t.Errorf("StringToJson(%#q).Len() => %v, want %v", mapstr, cnt,
		         len(jsontests));
	}
	for k,v := range values {
		if v1 := mapv.Get(k); !Equal(v1, v) {
			t.Errorf("MapTest: Walk(%#q) => %v, want %v", k, v1, v);
		}
	}
}
