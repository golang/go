// +build go1.12

package unitchecker

import "go/importer"

func init() {
	importerForCompiler = importer.ForCompiler
}
