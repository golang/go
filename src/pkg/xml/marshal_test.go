// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package xml

import (
	"reflect"
	"testing"

	"os"
	"bytes"
	"strings"
	"strconv"
)

type DriveType int

const (
	HyperDrive DriveType = iota
	ImprobabilityDrive
)

type Passenger struct {
	Name   []string `xml:"name"`
	Weight float32  `xml:"weight"`
}

type Ship struct {
	XMLName Name `xml:"spaceship"`

	Name      string       `xml:"attr"`
	Pilot     string       `xml:"attr"`
	Drive     DriveType    `xml:"drive"`
	Age       uint         `xml:"age"`
	Passenger []*Passenger `xml:"passenger"`
	secret    string
}

type RawXML string

func (rx RawXML) MarshalXML() ([]byte, os.Error) {
	return []byte(rx), nil
}

type NamedType string

type Port struct {
	XMLName Name   `xml:"port"`
	Type    string `xml:"attr"`
	Number  string `xml:"chardata"`
}

type Domain struct {
	XMLName Name   `xml:"domain"`
	Country string `xml:"attr"`
	Name    []byte `xml:"chardata"`
}

type Book struct {
	XMLName Name   `xml:"book"`
	Title   string `xml:"chardata"`
}

type SecretAgent struct {
	XMLName   Name   `xml:"agent"`
	Handle    string `xml:"attr"`
	Identity  string
	Obfuscate string `xml:"innerxml"`
}

type NestedItems struct {
	XMLName Name     `xml:"result"`
	Items   []string `xml:">item"`
	Item1   []string `xml:"Items>item1"`
}

type NestedOrder struct {
	XMLName Name   `xml:"result"`
	Field1  string `xml:"parent>c"`
	Field2  string `xml:"parent>b"`
	Field3  string `xml:"parent>a"`
}

type MixedNested struct {
	XMLName Name   `xml:"result"`
	A       string `xml:"parent1>a"`
	B       string `xml:"b"`
	C       string `xml:"parent1>parent2>c"`
	D       string `xml:"parent1>d"`
}

type NilTest struct {
	A interface{} `xml:"parent1>parent2>a"`
	B interface{} `xml:"parent1>b"`
	C interface{} `xml:"parent1>parent2>c"`
}

type Service struct {
	XMLName Name    `xml:"service"`
	Domain  *Domain `xml:"host>domain"`
	Port    *Port   `xml:"host>port"`
	Extra1  interface{}
	Extra2  interface{} `xml:"host>extra2"`
}

var nilStruct *Ship

var marshalTests = []struct {
	Value     interface{}
	ExpectXML string
}{
	// Test nil marshals to nothing
	{Value: nil, ExpectXML: ``},
	{Value: nilStruct, ExpectXML: ``},

	// Test value types (no tag name, so ???)
	{Value: true, ExpectXML: `<???>true</???>`},
	{Value: int(42), ExpectXML: `<???>42</???>`},
	{Value: int8(42), ExpectXML: `<???>42</???>`},
	{Value: int16(42), ExpectXML: `<???>42</???>`},
	{Value: int32(42), ExpectXML: `<???>42</???>`},
	{Value: uint(42), ExpectXML: `<???>42</???>`},
	{Value: uint8(42), ExpectXML: `<???>42</???>`},
	{Value: uint16(42), ExpectXML: `<???>42</???>`},
	{Value: uint32(42), ExpectXML: `<???>42</???>`},
	{Value: float32(1.25), ExpectXML: `<???>1.25</???>`},
	{Value: float64(1.25), ExpectXML: `<???>1.25</???>`},
	{Value: uintptr(0xFFDD), ExpectXML: `<???>65501</???>`},
	{Value: "gopher", ExpectXML: `<???>gopher</???>`},
	{Value: []byte("gopher"), ExpectXML: `<???>gopher</???>`},
	{Value: "</>", ExpectXML: `<???>&lt;/&gt;</???>`},
	{Value: []byte("</>"), ExpectXML: `<???>&lt;/&gt;</???>`},
	{Value: [3]byte{'<', '/', '>'}, ExpectXML: `<???>&lt;/&gt;</???>`},
	{Value: NamedType("potato"), ExpectXML: `<???>potato</???>`},
	{Value: []int{1, 2, 3}, ExpectXML: `<???>1</???><???>2</???><???>3</???>`},
	{Value: [3]int{1, 2, 3}, ExpectXML: `<???>1</???><???>2</???><???>3</???>`},

	// Test innerxml
	{Value: RawXML("</>"), ExpectXML: `</>`},
	{
		Value: &SecretAgent{
			Handle:    "007",
			Identity:  "James Bond",
			Obfuscate: "<redacted/>",
		},
		//ExpectXML: `<agent handle="007"><redacted/></agent>`,
		ExpectXML: `<agent handle="007"><Identity>James Bond</Identity><redacted/></agent>`,
	},

	// Test structs
	{Value: &Port{Type: "ssl", Number: "443"}, ExpectXML: `<port type="ssl">443</port>`},
	{Value: &Port{Number: "443"}, ExpectXML: `<port>443</port>`},
	{Value: &Port{Type: "<unix>"}, ExpectXML: `<port type="&lt;unix&gt;"></port>`},
	{Value: &Domain{Name: []byte("google.com&friends")}, ExpectXML: `<domain>google.com&amp;friends</domain>`},
	{Value: &Book{Title: "Pride & Prejudice"}, ExpectXML: `<book>Pride &amp; Prejudice</book>`},
	{Value: atomValue, ExpectXML: atomXml},
	{
		Value: &Ship{
			Name:  "Heart of Gold",
			Pilot: "Computer",
			Age:   1,
			Drive: ImprobabilityDrive,
			Passenger: []*Passenger{
				&Passenger{
					Name:   []string{"Zaphod", "Beeblebrox"},
					Weight: 7.25,
				},
				&Passenger{
					Name:   []string{"Trisha", "McMillen"},
					Weight: 5.5,
				},
				&Passenger{
					Name:   []string{"Ford", "Prefect"},
					Weight: 7,
				},
				&Passenger{
					Name:   []string{"Arthur", "Dent"},
					Weight: 6.75,
				},
			},
		},
		ExpectXML: `<spaceship name="Heart of Gold" pilot="Computer">` +
			`<drive>` + strconv.Itoa(int(ImprobabilityDrive)) + `</drive>` +
			`<age>1</age>` +
			`<passenger>` +
			`<name>Zaphod</name>` +
			`<name>Beeblebrox</name>` +
			`<weight>7.25</weight>` +
			`</passenger>` +
			`<passenger>` +
			`<name>Trisha</name>` +
			`<name>McMillen</name>` +
			`<weight>5.5</weight>` +
			`</passenger>` +
			`<passenger>` +
			`<name>Ford</name>` +
			`<name>Prefect</name>` +
			`<weight>7</weight>` +
			`</passenger>` +
			`<passenger>` +
			`<name>Arthur</name>` +
			`<name>Dent</name>` +
			`<weight>6.75</weight>` +
			`</passenger>` +
			`</spaceship>`,
	},
	// Test a>b
	{
		Value: NestedItems{Items: []string{}, Item1: []string{}},
		ExpectXML: `<result>` +
			`<Items>` +
			`</Items>` +
			`</result>`,
	},
	{
		Value: NestedItems{Items: []string{}, Item1: []string{"A"}},
		ExpectXML: `<result>` +
			`<Items>` +
			`<item1>A</item1>` +
			`</Items>` +
			`</result>`,
	},
	{
		Value: NestedItems{Items: []string{"A", "B"}, Item1: []string{}},
		ExpectXML: `<result>` +
			`<Items>` +
			`<item>A</item>` +
			`<item>B</item>` +
			`</Items>` +
			`</result>`,
	},
	{
		Value: NestedItems{Items: []string{"A", "B"}, Item1: []string{"C"}},
		ExpectXML: `<result>` +
			`<Items>` +
			`<item>A</item>` +
			`<item>B</item>` +
			`<item1>C</item1>` +
			`</Items>` +
			`</result>`,
	},
	{
		Value: NestedOrder{Field1: "C", Field2: "B", Field3: "A"},
		ExpectXML: `<result>` +
			`<parent>` +
			`<c>C</c>` +
			`<b>B</b>` +
			`<a>A</a>` +
			`</parent>` +
			`</result>`,
	},
	{
		Value: NilTest{A: "A", B: nil, C: "C"},
		ExpectXML: `<???>` +
			`<parent1>` +
			`<parent2><a>A</a></parent2>` +
			`<parent2><c>C</c></parent2>` +
			`</parent1>` +
			`</???>`,
	},
	{
		Value: MixedNested{A: "A", B: "B", C: "C", D: "D"},
		ExpectXML: `<result>` +
			`<parent1><a>A</a></parent1>` +
			`<b>B</b>` +
			`<parent1>` +
			`<parent2><c>C</c></parent2>` +
			`<d>D</d>` +
			`</parent1>` +
			`</result>`,
	},
	{
		Value:     Service{Port: &Port{Number: "80"}},
		ExpectXML: `<service><host><port>80</port></host></service>`,
	},
	{
		Value:     Service{},
		ExpectXML: `<service></service>`,
	},
	{
		Value: Service{Port: &Port{Number: "80"}, Extra1: "A", Extra2: "B"},
		ExpectXML: `<service>` +
			`<host><port>80</port></host>` +
			`<Extra1>A</Extra1>` +
			`<host><extra2>B</extra2></host>` +
			`</service>`,
	},
	{
		Value: Service{Port: &Port{Number: "80"}, Extra2: "example"},
		ExpectXML: `<service>` +
			`<host><port>80</port></host>` +
			`<host><extra2>example</extra2></host>` +
			`</service>`,
	},
}

func TestMarshal(t *testing.T) {
	for idx, test := range marshalTests {
		buf := bytes.NewBuffer(nil)
		err := Marshal(buf, test.Value)
		if err != nil {
			t.Errorf("#%d: Error: %s", idx, err)
			continue
		}
		if got, want := buf.String(), test.ExpectXML; got != want {
			if strings.Contains(want, "\n") {
				t.Errorf("#%d: marshal(%#v) - GOT:\n%s\nWANT:\n%s", idx, test.Value, got, want)
			} else {
				t.Errorf("#%d: marshal(%#v) = %#q want %#q", idx, test.Value, got, want)
			}
		}
	}
}

var marshalErrorTests = []struct {
	Value      interface{}
	ExpectErr  string
	ExpectKind reflect.Kind
}{
	{
		Value:      make(chan bool),
		ExpectErr:  "xml: unsupported type: chan bool",
		ExpectKind: reflect.Chan,
	},
	{
		Value: map[string]string{
			"question": "What do you get when you multiply six by nine?",
			"answer":   "42",
		},
		ExpectErr:  "xml: unsupported type: map[string] string",
		ExpectKind: reflect.Map,
	},
	{
		Value:      map[*Ship]bool{nil: false},
		ExpectErr:  "xml: unsupported type: map[*xml.Ship] bool",
		ExpectKind: reflect.Map,
	},
}

func TestMarshalErrors(t *testing.T) {
	for idx, test := range marshalErrorTests {
		buf := bytes.NewBuffer(nil)
		err := Marshal(buf, test.Value)
		if got, want := err, test.ExpectErr; got == nil {
			t.Errorf("#%d: want error %s", idx, want)
			continue
		} else if got.String() != want {
			t.Errorf("#%d: marshal(%#v) = [error] %q, want %q", idx, test.Value, got, want)
		}
		if got, want := err.(*UnsupportedTypeError).Type.Kind(), test.ExpectKind; got != want {
			t.Errorf("#%d: marshal(%#v) = [error kind] %s, want %s", idx, test.Value, got, want)
		}
	}
}

// Do invertibility testing on the various structures that we test
func TestUnmarshal(t *testing.T) {
	for i, test := range marshalTests {
		// Skip the nil pointers
		if i <= 1 {
			continue
		}

		var dest interface{}

		switch test.Value.(type) {
		case *Ship, Ship:
			dest = &Ship{}
		case *Port, Port:
			dest = &Port{}
		case *Domain, Domain:
			dest = &Domain{}
		case *Feed, Feed:
			dest = &Feed{}
		default:
			continue
		}

		buffer := bytes.NewBufferString(test.ExpectXML)
		err := Unmarshal(buffer, dest)

		// Don't compare XMLNames
		switch fix := dest.(type) {
		case *Ship:
			fix.XMLName = Name{}
		case *Port:
			fix.XMLName = Name{}
		case *Domain:
			fix.XMLName = Name{}
		case *Feed:
			fix.XMLName = Name{}
			fix.Author.InnerXML = ""
			for i := range fix.Entry {
				fix.Entry[i].Author.InnerXML = ""
			}
		}

		if err != nil {
			t.Errorf("#%d: unexpected error: %#v", i, err)
		} else if got, want := dest, test.Value; !reflect.DeepEqual(got, want) {
			t.Errorf("#%d: unmarshal(%#s) = %#v, want %#v", i, test.ExpectXML, got, want)
		}
	}
}

func BenchmarkMarshal(b *testing.B) {
	idx := len(marshalTests) - 1
	test := marshalTests[idx]

	buf := bytes.NewBuffer(nil)
	for i := 0; i < b.N; i++ {
		Marshal(buf, test.Value)
		buf.Truncate(0)
	}
}

func BenchmarkUnmarshal(b *testing.B) {
	idx := len(marshalTests) - 1
	test := marshalTests[idx]
	sm := &Ship{}
	xml := []byte(test.ExpectXML)

	for i := 0; i < b.N; i++ {
		buffer := bytes.NewBuffer(xml)
		Unmarshal(buffer, sm)
	}
}
