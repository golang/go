package main

import (
	"bytes"
	"flag"
	"go/doc"
	"go/parser"
	"go/token"
	"log"
	"os"
	"path/filepath"
	"strings"
)

func isGoFile(fi os.FileInfo) bool {
	return strings.HasSuffix(fi.Name(), ".go") &&
		!strings.HasSuffix(fi.Name(), "_test.go")
}

func main() {
	fset := token.NewFileSet()
	rootDir := flag.String("root", "./", "root of filesystem tree to scan")
	flag.Parse()
	err := filepath.Walk(*rootDir, func(path string, fi os.FileInfo, err error) error {
		if !fi.IsDir() {
			return nil
		}
		pkgs, err := parser.ParseDir(fset, path, isGoFile, parser.ParseComments)
		if err != nil {
			log.Println(path, err)
			return nil
		}
		for _, pkg := range pkgs {
			d := doc.NewPackageDoc(pkg, path)
			buf := new(bytes.Buffer)
			doc.ToHTML(buf, []byte(d.Doc), nil)
			b := buf.Bytes()
			for {
				i := bytes.Index(b, []byte("<h3>"))
				if i == -1 {
					break
				}
				line := bytes.SplitN(b[i:], []byte("\n"), 2)[0]
				log.Printf("%s: %s", path, line)
				b = b[i+len(line):]
			}
		}
		return nil
	})
	if err != nil {
		log.Fatal(err)
	}
}
