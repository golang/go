// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package xml_test

import (
	"encoding/xml"
	"fmt"
	"os"
)

//	<person id="13">
//		<name>
//			<first>John</first>
//			<last>Doe</last>
//		</name>
//		<age>42</age>
//		<Married>false</Married>
//		<!-- Need more fields. -->
//	</person>
func ExampleMarshalIndent() {
	type Person struct {
		XMLName   xml.Name `xml:"person"`
		Id        int      `xml:"id,attr"`
		FirstName string   `xml:"name>first"`
		LastName  string   `xml:"name>last"`
		Age       int      `xml:"age"`
		Height    float32  `xml:"height,omitempty"`
		Married   bool
		Comment   string `xml:",comment"`
	}

	v := &Person{Id: 13, FirstName: "John", LastName: "Doe", Age: 42}
	v.Comment = " Need more fields. "

	output, err := xml.MarshalIndent(v, "\t", "\t")
	if err != nil {
		fmt.Printf("error: %v\n", err)
	}

	os.Stdout.Write(output)
}
