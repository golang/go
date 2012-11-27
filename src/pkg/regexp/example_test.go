package regexp_test

import (
	"fmt"
	"regexp"
)

func Example() {
	// Compile the expression once, usually at init time.
	// Use raw strings to avoid having to quote the backslashes.
	var validID = regexp.MustCompile(`^[a-z]+\[[0-9]+\]$`)

	fmt.Println(validID.MatchString("adam[23]"))
	fmt.Println(validID.MatchString("eve[7]"))
	fmt.Println(validID.MatchString("Job[48]"))
	fmt.Println(validID.MatchString("snakey"))
	// Output:
	// true
	// true
	// false
	// false
}

func ExampleMatchString() {
	matched, err := regexp.MatchString("foo.*", "seafood")
	fmt.Println(matched, err)
	matched, err = regexp.MatchString("bar.*", "seafood")
	fmt.Println(matched, err)
	matched, err = regexp.MatchString("a(b", "seafood")
	fmt.Println(matched, err)
	// Output:
	// true <nil>
	// false <nil>
	// false error parsing regexp: missing closing ): `a(b`
}

func ExampleRegexp_FindString() {
	re := regexp.MustCompile("fo.?")
	fmt.Printf("%q\n", re.FindString("seafood"))
	fmt.Printf("%q\n", re.FindString("meat"))
	// Output:
	// "foo"
	// ""
}

func ExampleRegexp_FindStringIndex() {
	re := regexp.MustCompile("ab?")
	fmt.Println(re.FindStringIndex("tablett"))
	fmt.Println(re.FindStringIndex("foo") == nil)
	// Output:
	// [1 3]
	// true
}

func ExampleRegexp_FindStringSubmatch() {
	re := regexp.MustCompile("a(x*)b(y|z)c")
	fmt.Printf("%q\n", re.FindStringSubmatch("-axxxbyc-"))
	fmt.Printf("%q\n", re.FindStringSubmatch("-abzc-"))
	// Output:
	// ["axxxbyc" "xxx" "y"]
	// ["abzc" "" "z"]
}

func ExampleRegexp_FindAllString() {
	re := regexp.MustCompile("a.")
	fmt.Println(re.FindAllString("paranormal", -1))
	fmt.Println(re.FindAllString("paranormal", 2))
	fmt.Println(re.FindAllString("graal", -1))
	fmt.Println(re.FindAllString("none", -1))
	// Output:
	// [ar an al]
	// [ar an]
	// [aa]
	// []
}

func ExampleRegexp_FindAllStringSubmatch() {
	re := regexp.MustCompile("a(x*)b")
	fmt.Printf("%q\n", re.FindAllStringSubmatch("-ab-", -1))
	fmt.Printf("%q\n", re.FindAllStringSubmatch("-axxb-", -1))
	fmt.Printf("%q\n", re.FindAllStringSubmatch("-ab-axb-", -1))
	fmt.Printf("%q\n", re.FindAllStringSubmatch("-axxb-ab-", -1))
	// Output:
	// [["ab" ""]]
	// [["axxb" "xx"]]
	// [["ab" ""] ["axb" "x"]]
	// [["axxb" "xx"] ["ab" ""]]
}

func ExampleRegexp_FindAllStringSubmatchIndex() {
	re := regexp.MustCompile("a(x*)b")
	// Indices:
	//    01234567   012345678
	//    -ab-axb-   -axxb-ab-
	fmt.Println(re.FindAllStringSubmatchIndex("-ab-", -1))
	fmt.Println(re.FindAllStringSubmatchIndex("-axxb-", -1))
	fmt.Println(re.FindAllStringSubmatchIndex("-ab-axb-", -1))
	fmt.Println(re.FindAllStringSubmatchIndex("-axxb-ab-", -1))
	fmt.Println(re.FindAllStringSubmatchIndex("-foo-", -1))
	// Output:
	// [[1 3 2 2]]
	// [[1 5 2 4]]
	// [[1 3 2 2] [4 7 5 6]]
	// [[1 5 2 4] [6 8 7 7]]
	// []
}

func ExampleRegexp_ReplaceAllLiteralString() {
	re := regexp.MustCompile("a(x*)b")
	fmt.Println(re.ReplaceAllLiteralString("-ab-axxb-", "T"))
	fmt.Println(re.ReplaceAllLiteralString("-ab-axxb-", "$1"))
	fmt.Println(re.ReplaceAllLiteralString("-ab-axxb-", "${1}"))
	// Output:
	// -T-T-
	// -$1-$1-
	// -${1}-${1}-
}

func ExampleRegexp_ReplaceAllString() {
	re := regexp.MustCompile("a(x*)b")
	fmt.Println(re.ReplaceAllString("-ab-axxb-", "T"))
	fmt.Println(re.ReplaceAllString("-ab-axxb-", "$1"))
	fmt.Println(re.ReplaceAllString("-ab-axxb-", "$1W"))
	fmt.Println(re.ReplaceAllString("-ab-axxb-", "${1}W"))
	// Output:
	// -T-T-
	// --xx-
	// ---
	// -W-xxW-
}

func ExampleRegexp_SubexpNames() {
	re := regexp.MustCompile("(?P<first>[a-zA-Z]+) (?P<last>[a-zA-Z]+)")
	fmt.Println(re.MatchString("Alan Turing"))
	fmt.Printf("%q\n", re.SubexpNames())
	reversed := fmt.Sprintf("${%s} ${%s}", re.SubexpNames()[2], re.SubexpNames()[1])
	fmt.Println(reversed)
	fmt.Println(re.ReplaceAllString("Alan Turing", reversed))
	// Output:
	// true
	// ["" "first" "last"]
	// ${last} ${first}
	// Turing Alan
}
