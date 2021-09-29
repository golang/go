package typeparams

import (
	"encoding/json"
	"fmt"
)

func unmarshalT[T any](data []byte) T {
	var x T
	json.Unmarshal(data, x)
	return x
}

func unmarshalT2[T any](data []byte, t T) {
    json.Unmarshal(data, t)
}

func main() {
	x := make(map[string]interface{})
	unmarshalT2([]byte(`{"a":1}`), &x)
	fmt.Println(x)
}