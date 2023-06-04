// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package json

import (
	"testing"
)

func testEqual[T jsonType](t *testing.T, tt testType[T]) {

	var eq bool
	var err error

	if tt.sN == nil {
		eq, err = Eq(tt.s1, tt.s2)

		if err != nil {
			t.Errorf("%T:%d; err:%s\n", tt, tt.id, err)
			return
		}
	} else if tt.debug {
		eq, err = Equal(tt.s1, tt.s2, tt.sN...)
	} else {
		eq = DeeplyEqual(tt.s1, tt.s2, tt.sN...)
	}
	if err != nil {
		t.Errorf("%T:%d; err:%s\n", tt, tt.id, err)
		return
	}

	if eq != tt.want {
		t.Errorf("%T:%d; want: %t got: %t\n", tt, tt.id, tt.want, eq)
	}
}

func TestEqual(t *testing.T) {

	byteCases := getTestcases[[]byte]()

	for _, tt := range byteCases {
		testEqual(t, tt)
	}

	strCases := getTestcases[string]()

	for _, tt := range strCases {
		testEqual(t, tt)
	}

}

type testType[T jsonType] struct {
	want  bool
	s1    T
	s2    T
	sN    []T
	id    int
	debug bool
}

func getTestcases[T jsonType]() (testcases []testType[T]) {
	ipTestcases := []testType[string]{
		{
			want: true,
			s1:   "2",
			s2:   "2",
		},
		{
			want: true,
			s1:   "[1,2,3]",
			s2:   "[1,2,3]",
		},
		{
			want: false,
			s1:   "true",
			s2:   "false",
		},
		{
			want: true,
			s1:   `{"name":"Hiro","email":"laciferin@gmail.com","age":19}`,
			s2:   `{"name":"Hiro","age":19,"email":"laciferin@gmail.com"}`,
			sN: []string{
				`{"age":19,"name":"Hiro","email":"laciferin@gmail.com"}`,
			},
		},
		{
			want: true,
			s1:   `{"key1":"value1","key2":"value2"}`,
			s2:   `{"key2":"value2","key1":"value1"}`,
			sN: []string{
				`{"key1":"value1","key2":"value2"}`,
				`{"key2":"value2","key1":"value1"}`,
			},
		},
		{
			want: false,
			s1:   `{"name":"Alice","age":25}`,
			s2:   `{"name":"Bob","age":25}`,
			sN: []string{
				`{"name":"Charlie","age":25}`,
				`{"name":"David","age":25}`,
			},
		},
		{
			want: true,
			s1: `{
			"name": "Hiro",
			"age": 10,
			"address": {
				"street": "123 Main St",
				"city": "New York",
				"state": "NY"
				}
			}`,
			s2: `
				{
				  "name": "Hiro",
				  "age": 10,
				  "address": {
					"city": "New York",
					"state": "NY",
					"street": "123 Main St"
				  }
				}
			`,
		},
		{
			want: false, //Structure, order fails for company.employees:arr
			s1:   `{"person":{"name":"Hiro","age":19,"address":{"street":"456 Elm St","city":"Los Angeles","state":"CA"}},"company":{"name":"ABC Corporation","location":"San Francisco","employees":[100,200,300]}}`,
			s2:   `{"company":{"employees":[100,300,200],"name":"ABC Corporation","location":"San Francisco"},"person":{"age":19,"address":{"state":"CA","city":"Los Angeles","street":"456 Elm St"},"name":"Hiro"}}`,
		},
		{
			want: true, //same as [ipTestcases][4] but order maintained for company.employees
			s1:   `{"person":{"name":"Hiro","age":19,"address":{"street":"456 Elm St","city":"Los Angeles","state":"CA"}},"company":{"name":"ABC Corporation","location":"San Francisco","employees":[100,200,300]}}`,
			s2:   `{"company":{"employees":[100,200,300],"name":"ABC Corporation","location":"San Francisco"},"person":{"age":19,"address":{"state":"CA","city":"Los Angeles","street":"456 Elm St"},"name":"Hiro"}}`,
		},
		{
			want: true, //[ipTestcases][4].sN++
			s1:   `{"person":{"name":"Hiro","age":19,"address":{"street":"456 Elm St","city":"Los Angeles","state":"CA"}},"company":{"name":"ABC Corporation","location":"San Francisco","employees":[100,200,300]}}`,
			s2:   `{"company":{"employees":[100,200,300],"name":"ABC Corporation","location":"San Francisco"},"person":{"age":19,"address":{"state":"CA","city":"Los Angeles","street":"456 Elm St"},"name":"Hiro"}}`,
			sN: []string{
				`{
				"person": {
					"name": "Hiro",
					"age": 19,
					"address": {
						"street": "456 Elm St",
						"city": "Los Angeles",
						"state": "CA"
					}
				},
				"company": {
					"name": "ABC Corporation",
					"location": "San Francisco",
					"employees": [100, 200, 300]
				}
			}`,
				`{"company":{"employees":[100,200,300],"name":"ABC Corporation","location":"San Francisco"},"person":{"age":19,"address":{"state":"CA","city":"Los Angeles","street":"456 Elm St"},"name":"Hiro"}}`,
			},
		},
	}

	for i, ip := range ipTestcases {

		testcase := testType[T]{
			want:  ip.want,
			s1:    T(ip.s1),
			s2:    T(ip.s2),
			debug: true,
			id:    i + 1,
		}

		for _, sI := range ip.sN {
			testcase.sN = append(testcase.sN, T(sI))
		}

		testcases = append(testcases, testcase)

	}

	return
}
