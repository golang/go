package json

import (
	"fmt"
	"testing"
)

func TestEqual(t *testing.T) {
	for _, tt := range getTests() {
		eq, err := Equal(tt.leftJSON, tt.rightJSON)
		if eq != tt.want {
			t.Errorf("should eq \nleftJSON:%s \nrightJSON:%s", tt.leftJSON, tt.rightJSON)
		}
		if err != nil {
			panic(err)
		}
		eq, err = Equal(tt.leftJSON, tt.errJSON)
		if err != nil {
			panic(err)
		}
		if eq {
			t.Errorf("should !eq leftJSON:%s errJSON:%s", tt.leftJSON, tt.errJSON)
		}
	}
}

type testEqual struct {
	leftJSON  string
	rightJSON string
	want      bool
	errJSON   string
}

func ExampleEqual() {
	json1 := `{"author":"北洛","country":"china","age":22,"gopher":true}`
	json2 := `{"author":"北洛","gopher":true,"country":"china","age":22}`
	json3 := `{"age":22,"pageNum":1,"author":"北洛"}`
	json4 := `{"age":22,"pageNum":1,"author":"洛北"}`
	json5 := `{"age":22,"pageNum":1}`

	fmt.Println(Equal(json1, json2))        //Returns true if the structure is equal and the values are equal
	fmt.Println(Equal(json1, json2, json3)) //To return true, multiple values must be evaluated and all values must be equivalent
	fmt.Println(Equal(json3, json4))        //Structure is equal but values are not equal, not equivalent
	fmt.Println(Equal(json3, json5))        //Returns false due to unequal structures
	//Output:
	//true <nil>
	//false <nil>
	//false <nil>
	//false <nil>

}
func getTests() []testEqual {
	tests := []testEqual{
		{
			leftJSON:  `{"A":[{"name":"tag"}]}`,
			rightJSON: `{"A":[{"name":"tag"}]}`,
			want:      true,
			errJSON:   `{"A":[{"name":"tag"},{"name":"tag"}]}`,
		}, {
			leftJSON:  `{"B":[{"name":"tag"},{"name":"tag"}]}`,
			rightJSON: `{"B":[{"name":"tag"},{"name":"tag"}]}`,
			want:      true,
			errJSON:   `{"C":[{"name":"tag"},{"name":"tag"}]}`,
		}, {
			leftJSON:  `{"C":[{"name":"tag"},{"name":"tag"},{"name":"tag"}]}`,
			rightJSON: `{"C":[{"name":"tag"},{"name":"tag"},{"name":"tag"}]}`,
			want:      true,
			errJSON:   `{"C":[{"name":"tag"},{"name":"tag"},{"name":"tag"}],"gopher":true}`,
		}, {
			leftJSON:  `{"AP":[{"name":"tag"}]}`,
			rightJSON: `{"AP":[{"name":"tag"}]}`,
			want:      true,
			errJSON:   `{"AP":[{"name":"北洛"}]}`,
		}, {
			leftJSON:  `{"BP":[{"name":"tag"},{"name":"tag"}]}`,
			rightJSON: `{"BP":[{"name":"tag"},{"name":"tag"}]}`,
			want:      true,
			errJSON:   `{"BP":[{"name":"tag"},{"name":"china"}]}`,
		}, {
			leftJSON:  `{"AP":[{"name":"tag"}],"APP":[{"name":"tag"}],"B":[{"name":"tag"},{"name":"tag"}],"BP":[{"name":"tag"},{"name":"tag"}],"BPP":[{"name":"tag"},{"name":"tag"}],"C":[{"name":"tag"},{"name":"tag"},{"name":"tag"}],"CP":[{"name":"tag"},{"name":"tag"},{"name":"tag"}],"CPP":[{"name":"tag"},{"name":"tag"},{"name":"tag"}]}`,
			rightJSON: `{"APP":[{"name":"tag"}],"CPP":[{"name":"tag"},{"name":"tag"},{"name":"tag"}],"AP":[{"name":"tag"}],"B":[{"name":"tag"},{"name":"tag"}],"BP":[{"name":"tag"},{"name":"tag"}],"BPP":[{"name":"tag"},{"name":"tag"}],"C":[{"name":"tag"},{"name":"tag"},{"name":"tag"}],"CP":[{"name":"tag"},{"name":"tag"},{"name":"tag"}]}`,
			want:      true,
			errJSON:   `{"APP":[{"name":"tag"},{"name":"tag"}],"CPP":[{"name":"tag"},{"name":"tag"},{"name":"tag"}],"AP":[{"name":"tag"}],"B":[{"name":"tag"},{"name":"tag"}],"BP":[{"name":"tag"},{"name":"tag"}],"BPP":[{"name":"tag"},{"name":"tag"}],"C":[{"name":"tag"},{"name":"tag"},{"name":"tag"}],"CP":[{"name":"tag"},{"name":"tag"},{"name":"tag"}]}`,
		}, {
			leftJSON:  `{"AP":[{"name":"tag"}],"APP":[{"name":"tag"}],"B":[{"name":"tag"},{"name":"tag"}],"BP":[{"name":"tag"},{"name":"tag"}],"BPP":[{"name":"tag"},{"name":"tag"}],"C":[{"name":"tag"},{"name":"tag"},{"name":"tag"}],"CP":[{"name":"tag"},{"name":"tag"},{"name":"tag"}],"CPP":[{"name":"tag"},{"name":"tag"},{"name":"tag"}]}`,
			rightJSON: `{"B":[{"name":"tag"},{"name":"tag"}],"APP":[{"name":"tag"}],"CPP":[{"name":"tag"},{"name":"tag"},{"name":"tag"}],"AP":[{"name":"tag"}],"BP":[{"name":"tag"},{"name":"tag"}],"BPP":[{"name":"tag"},{"name":"tag"}],"C":[{"name":"tag"},{"name":"tag"},{"name":"tag"}],"CP":[{"name":"tag"},{"name":"tag"},{"name":"tag"}]}`,
			want:      true,
			errJSON:   `{"APP":[{"name":"tag1"}],"CPP":[{"name":"tag"},{"name":"tag"},{"name":"tag"}],"AP":[{"name":"tag"}],"B":[{"name":"tag"},{"name":"tag"}],"BP":[{"name":"tag"},{"name":"tag"}],"BPP":[{"name":"tag"},{"name":"tag"}],"C":[{"name":"tag"},{"name":"tag"},{"name":"tag"}],"CP":[{"name":"tag"},{"name":"tag"},{"name":"tag"}]}`,
		},
		{
			leftJSON:  `{"AP":[{"name":"tag"}],"APP":[{"name":"tag"}],"B":[{"name":"tag"},{"name":"tag"}],"BP":[{"name":"tag"},{"name":"tag"}],"BPP":[{"name":"tag"},{"name":"tag"}],"C":[{"name":"tag"},{"name":"tag"},{"name":"tag"}],"CP":[{"name":"tag"},{"name":"tag"},{"name":"tag"}],"CPP":[{"name":"tag"},{"name":"tag"},{"name":"tag"}]}`,
			rightJSON: `{"B":[{"name":"tag"},{"name":"tag"}],"APP":[{"name":"tag"}],"CPP":[{"name":"tag"},{"name":"tag"},{"name":"tag"}],"AP":[{"name":"tag"}],"BP":[{"name":"tag"},{"name":"tag"}],"BPP":[{"name":"tag"},{"name":"tag"}],"C":[{"name":"tag"},{"name":"tag"},{"name":"tag"}],"CP":[{"name":"tag"},{"name":"tag"},{"name":"tag"}]}`,
			want:      true,
			errJSON:   `{"APP":[{"name1":"tag"}],"CPP":[{"name":"tag"},{"name":"tag"},{"name":"tag"}],"AP":[{"name":"tag"}],"B":[{"name":"tag"},{"name":"tag"}],"BP":[{"name":"tag"},{"name":"tag"}],"BPP":[{"name":"tag"},{"name":"tag"}],"C":[{"name":"tag"},{"name":"tag"},{"name":"tag"}],"CP":[{"name":"tag"},{"name":"tag"},{"name":"tag"}]}`,
		}, {
			leftJSON:  `{"langAge":[{"arts":[{"profile":{"c":"clang"},"values":["1","2"]}]},{"arts":[{"profile":{"c++":"cpp"},"values":["cpp1","cpp2"]}]},{"arts":[{"profile":{"Golang":"go"},"values":["Golang","Golang1"]}]}],"uid":1}`,
			rightJSON: `{"langAge":[{"arts":[{"values":["1","2"],"profile":{"c":"clang"}}]},{"arts":[{"profile":{"c++":"cpp"},"values":["cpp1","cpp2"]}]},{"arts":[{"profile":{"Golang":"go"},"values":["Golang","Golang1"]}]}],"uid":1}`,
			want:      true,
			errJSON:   `{"langAge":[{"arts":[{"values":["11 there","2"],"profile":{"c":"clang"}}]},{"arts":[{"profile":{"c++":"cpp"},"values":["cpp1","cpp2"]}]},{"arts":[{"profile":{"Golang":"go"},"values":["Golang","Golang1"]}]}],"uid":1}`,
		}, {
			leftJSON:  `{"langAge":[{"arts":[{"values":["1","2"],"profile":{"c":"clang"}}]},{"arts":[{"profile":{"c++":"cpp"},"values":["cpp1","cpp2"]}]},{"arts":[{"values":["Golang","Golang1"],"profile":{"Golang":"go"}}]}],"uid":1}`,
			rightJSON: `{"langAge":[{"arts":[{"values":["1","2"],"profile":{"c":"clang"}}]},{"arts":[{"profile":{"c++":"cpp"},"values":["cpp1","cpp2"]}]},{"arts":[{"profile":{"Golang":"go"},"values":["Golang","Golang1"]}]}],"uid":1}`,
			want:      true,
			errJSON:   `{"langAge":[{"arts_there":[{"values":["11 there","2"],"profile":{"c":"clang"}}]},{"arts":[{"profile":{"c++":"cpp"},"values":["cpp1","cpp2"]}]},{"arts":[{"profile":{"Golang":"go"},"values":["Golang","Golang1"]}]}],"uid":1}`,
		}, {
			leftJSON:  `{"langAge":[{"arts":[{"values":["1","2"],"profile":{"c":"clang"}}]},{"arts":[{"profile":{"c++":"cpp"},"values":["cpp1","cpp2"]}]},{"arts":[{"values":["Golang","Golang1"],"profile":{"Golang":"go"}}]}],"uid":1}`,
			rightJSON: `{"langAge":[{"arts":[{"values":["1","2"],"profile":{"c":"clang"}}]},{"arts":[{"profile":{"c++":"cpp"},"values":["cpp1","cpp2"]}]},{"arts":[{"profile":{"Golang":"go"},"values":["Golang","Golang1"]}]}],"uid":1}`,
			want:      true,
			errJSON:   `{"langAge":[{"arts":[{"values":["1","2"],"profile":{"c":"clang"}}]},{"arts":[{"profile":{"c++":"cpp"},"values":["cpp1","cpp2","cpp3"]}]},{"arts":[{"profile":{"Golang":"go"},"values":["Golang","Golang1"]}]}],"uid":1}`,
		},
	}

	return tests
}
