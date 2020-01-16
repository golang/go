// Package packagesinternal exposes internal-only fields from go/packages.
package packagesinternal

var GetForTest = func(p interface{}) string { return "" }
