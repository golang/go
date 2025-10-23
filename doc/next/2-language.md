## Changes to the language {#language}

<!-- https://go.dev/issue/45624 --->

The built-in `new` function, which creates a new variable, now allows
its operand to be an expression, specifying the initial value of the
variable.

This feature is particularly useful when working with serialization
packages such as `encoding/json` or protocol buffers that use a
pointer to represent an optional value, as it enables an optional
field to be populated in a simple expression, for example:

```go
import "encoding/json"

type Person struct {
	Name string   `json:"name"`
	Age  *int     `json:"age"` // age if known; nil otherwise
}

func personJSON(name string, born time.Time) ([]byte, error) {
	return json.Marshal(Person{
		Name: name,
		Age:  new(yearsSince(born)),
	})
}

func yearsSince(t time.Time) int {
	return int(time.Since(t).Hours() / (365.25 * 24)) // approximately
}
```
