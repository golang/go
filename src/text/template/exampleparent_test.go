package template_test

import (
	"log"
	"os"
	"text/template"
)

func ExampleTemplate_parent() {
	const base_html = `<!DOCTYPE html>
<html>
	<head>
		<title>{{block "title" .}}My website{{end}}</title>
	</head>
	<body>
		<main>
			{{- block "content" .}}{{end -}}
		</main>
		<footer>
			{{- block "footer" .}}
			<p>Thanks for visiting!</p>
			{{- end}}
		</footer>
	</body>
</html>

`
	base := template.Must(template.New("base.html").Parse(base_html))

	const index_html = `{{define "content"}}<h1>Welcome!</h1>{{end}}`
	index := template.Must(template.Must(base.Clone()).New("index.html").Parse(index_html))

	{
		err := index.ExecuteTemplate(os.Stdout, "base.html", nil)
		if err != nil {
			log.Println("executing template:", err)
		}
	}

	const about_html = `{{define "title"}}{{template parent .}} - About{{end}}
{{define "content"}}<h1>About us</h1>{{end}}`
	about := template.Must(template.Must(base.Clone()).New("about.html").Parse(about_html))

	{
		err := about.ExecuteTemplate(os.Stdout, "base.html", nil)
		if err != nil {
			log.Println("executing template:", err)
		}
	}

	// Output:
	// <!DOCTYPE html>
	// <html>
	// 	<head>
	// 		<title>My website</title>
	// 	</head>
	// 	<body>
	// 		<main><h1>Welcome!</h1></main>
	// 		<footer>
	// 			<p>Thanks for visiting!</p>
	// 		</footer>
	// 	</body>
	// </html>
	//
	// <!DOCTYPE html>
	// <html>
	// 	<head>
	// 		<title>My website - About</title>
	// 	</head>
	// 	<body>
	// 		<main><h1>About us</h1></main>
	// 		<footer>
	// 			<p>Thanks for visiting!</p>
	// 		</footer>
	// 	</body>
	// </html>

}
