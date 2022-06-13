package inlayHint //@inlayHint("package")

import "fmt"

func fieldNames() {
	for _, c := range []struct {
		in, want string
	}{
		struct{ in, want string }{"Hello, world", "dlrow ,olleH"},
		{"Hello, 世界", "界世 ,olleH"},
		{"", ""},
	} {
		fmt.Println(c.in == c.want)
	}
}

func fieldNamesPointers() {
	for _, c := range []*struct {
		in, want string
	}{
		&struct{ in, want string }{"Hello, world", "dlrow ,olleH"},
		{"Hello, 世界", "界世 ,olleH"},
		{"", ""},
	} {
		fmt.Println(c.in == c.want)
	}
}
