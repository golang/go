// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package template_test

import (
	"fmt"
	"html/template"
	"log"
	"os"
	"strings"
)

func Example() {
	const tpl = `
<!DOCTYPE html>
<html>
	<head>
		<meta charset="UTF-8">
		<title>{{.Title}}</title>
	</head>
	<body>
		{{range .Items}}<div>{{ . }}</div>{{else}}<div><strong>no rows</strong></div>{{end}}
	</body>
</html>`

	check := func(err error) {
		if err != nil {
			log.Fatal(err)
		}
	}
	t, err := template.New("webpage").Parse(tpl)
	check(err)

	data := struct {
		Title string
		Items []string
	}{
		Title: "My page",
		Items: []string{
			"My photos",
			"My blog",
		},
	}

	err = t.Execute(os.Stdout, data)
	check(err)

	noItems := struct {
		Title string
		Items []string
	}{
		Title: "My another page",
		Items: []string{},
	}

	err = t.Execute(os.Stdout, noItems)
	check(err)

	// Output:
	// <!DOCTYPE html>
	// <html>
	// 	<head>
	// 		<meta charset="UTF-8">
	// 		<title>My page</title>
	// 	</head>
	// 	<body>
	// 		<div>My photos</div><div>My blog</div>
	// 	</body>
	// </html>
	// <!DOCTYPE html>
	// <html>
	// 	<head>
	// 		<meta charset="UTF-8">
	// 		<title>My another page</title>
	// 	</head>
	// 	<body>
	// 		<div><strong>no rows</strong></div>
	// 	</body>
	// </html>

}

func Example_autoescaping() {
	check := func(err error) {
		if err != nil {
			log.Fatal(err)
		}
	}
	t, err := template.New("foo").Parse(`{{define "T"}}Hello, {{.}}!{{end}}`)
	check(err)
	err = t.ExecuteTemplate(os.Stdout, "T", "<script>alert('you have been pwned')</script>")
	check(err)
	// Output:
	// Hello, &lt;script&gt;alert(&#39;you have been pwned&#39;)&lt;/script&gt;!
}

func Example_escape() {
	const s = `"Fran & Freddie's Diner" <tasty@example.com>`
	v := []interface{}{`"Fran & Freddie's Diner"`, ' ', `<tasty@example.com>`}

	fmt.Println(template.HTMLEscapeString(s))
	template.HTMLEscape(os.Stdout, []byte(s))
	fmt.Fprintln(os.Stdout, "")
	fmt.Println(template.HTMLEscaper(v...))

	fmt.Println(template.JSEscapeString(s))
	template.JSEscape(os.Stdout, []byte(s))
	fmt.Fprintln(os.Stdout, "")
	fmt.Println(template.JSEscaper(v...))

	fmt.Println(template.URLQueryEscaper(v...))

	// Output:
	// &#34;Fran &amp; Freddie&#39;s Diner&#34; &lt;tasty@example.com&gt;
	// &#34;Fran &amp; Freddie&#39;s Diner&#34; &lt;tasty@example.com&gt;
	// &#34;Fran &amp; Freddie&#39;s Diner&#34;32&lt;tasty@example.com&gt;
	// \"Fran \x26 Freddie\'s Diner\" \x3Ctasty@example.com\x3E
	// \"Fran \x26 Freddie\'s Diner\" \x3Ctasty@example.com\x3E
	// \"Fran \x26 Freddie\'s Diner\"32\x3Ctasty@example.com\x3E
	// %22Fran+%26+Freddie%27s+Diner%2232%3Ctasty%40example.com%3E

}

func ExampleTemplate_Delims() {
	const text = "<<.Greeting>> {{.Name}}"

	data := struct {
		Greeting string
		Name     string
	}{
		Greeting: "Hello",
		Name:     "Joe",
	}

	t := template.Must(template.New("tpl").Delims("<<", ">>").Parse(text))

	err := t.Execute(os.Stdout, data)
	if err != nil {
		log.Fatal(err)
	}

	// Output:
	// Hello {{.Name}}
}

// The following example is duplicated in text/template; keep them in sync.

func ExampleTemplate_block() {
	const (
		master  = `Names:{{block "list" .}}{{"\n"}}{{range .}}{{println "-" .}}{{end}}{{end}}`
		overlay = `{{define "list"}} {{join . ", "}}{{end}} `
	)
	var (
		funcs     = template.FuncMap{"join": strings.Join}
		guardians = []string{"Gamora", "Groot", "Nebula", "Rocket", "Star-Lord"}
	)
	masterTmpl, err := template.New("master").Funcs(funcs).Parse(master)
	if err != nil {
		log.Fatal(err)
	}
	overlayTmpl, err := template.Must(masterTmpl.Clone()).Parse(overlay)
	if err != nil {
		log.Fatal(err)
	}
	if err := masterTmpl.Execute(os.Stdout, guardians); err != nil {
		log.Fatal(err)
	}
	if err := overlayTmpl.Execute(os.Stdout, guardians); err != nil {
		log.Fatal(err)
	}
	// Output:
	// Names:
	// - Gamora
	// - Groot
	// - Nebula
	// - Rocket
	// - Star-Lord
	// Names: Gamora, Groot, Nebula, Rocket, Star-Lord
}
