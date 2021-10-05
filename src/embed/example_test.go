package embed_test

import (
	"embed"
	"log"
	"net/http"
)

//└── static
//    ├── apple.png
//    ├── banana.png
//    └── cake.png
//go:embed static/*.png
var images embed.FS

func Example() {
	mux := http.NewServeMux()
	mux.Handle("/static/", http.FileServer(http.FS(images)))
	err := http.ListenAndServe(":8080", mux)
	if err != nil {
		log.Fatal(err)
	}

	// curl localhost:8080/static/apple.png --output apple.png
	// curl localhost:8080/static/banana.png --output banana.png
	// curl localhost:8080/static/cake.png --output cake.png
	// Output:
}
